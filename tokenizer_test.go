package ollamatokenizer

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
)

const defaultModelsDir = "/var/lib/ollama"

func listModels(t *testing.T) []string {
	t.Helper()
	manifests, err := manifest.Manifests(true)
	if err != nil {
		t.Fatalf("listModels: %v", err)
	}
	names := make([]string, 0, len(manifests))
	for n := range manifests {
		names = append(names, n.DisplayShortest())
	}
	// OLLAMATOKENIZER_TEST_MODELS (comma-separated) restricts the set under test.
	if filter := os.Getenv("OLLAMATOKENIZER_TEST_MODELS"); filter != "" {
		want := strings.Split(filter, ",")
		allowed := make(map[string]bool, len(want))
		for _, w := range want {
			allowed[strings.TrimSpace(w)] = true
		}
		filtered := names[:0]
		for _, n := range names {
			if allowed[n] {
				filtered = append(filtered, n)
			}
		}
		names = filtered
	}
	return names
}

func ensureModelsDir(t *testing.T) {
	t.Helper()
	if os.Getenv("OLLAMA_MODELS") == "" {
		t.Setenv("OLLAMA_MODELS", defaultModelsDir)
	}
}

func ollamaURL(t *testing.T) string {
	t.Helper()
	url := os.Getenv("OLLAMA_HOST")
	if url == "" {
		url = "http://localhost:11434"
	}
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get(url + "/api/tags")
	if err != nil {
		t.Skipf("ollama server not reachable at %s: %v", url, err)
	}
	resp.Body.Close()
	return url
}

// String levels exercise thinking-descriptor resolution and legacy validation.
var thinkModes = []struct {
	name  string
	think *api.ThinkValue
}{
	{"think_nil", nil},
	{"think_true", &api.ThinkValue{Value: true}},
	{"think_false", &api.ThinkValue{Value: false}},
	{"think_medium", &api.ThinkValue{Value: "medium"}},
	{"think_high", &api.ThinkValue{Value: "high"}},
	{"think_low", &api.ThinkValue{Value: "low"}},
	{"think_max", &api.ThinkValue{Value: "max"}},
}

// testMessages exercises edge cases: unicode, special chars, think tags, all roles.
var testMessages = []api.Message{
	{Role: "system", Content: "You are a helpful assistant. Respond concisely. Use 中文 when asked."},
	{Role: "user", Content: "Hello, 世界! What is 2+2?"},
	{Role: "assistant", Content: "<think>Let me calculate 2+2</think>2+2=4. The answer is 4."},
	{Role: "user", Content: "café résumé naïve — \"quotes\" 'apos' <html> &amp;\n" +
		"emoji: 🌍🤖✅\n" +
		"code: func main() { fmt.Println(\"hi\") }\n" +
		"$100 @user #tag https://example.com/path?q=1&r=2"},
	{Role: "assistant", Content: "<think>Processing the complex input</think>Got it! Here's a summary:\n" +
		"\tLine one.\n" +
		"   Extra   spaces.\n" +
		"The URL is https://example.com/path?q=1&r=2"},
	{Role: "user", Content: "And what about 德国?"},
}

// testPrompt returns the last user message from testMessages.
func testPrompt() string {
	for i := len(testMessages) - 1; i >= 0; i-- {
		if testMessages[i].Role == "user" {
			return testMessages[i].Content
		}
	}
	return ""
}

// TestTokenizeGenerateMatchesAPI compares our tokens to live /api/generate.
// Count vs prompt_eval_count (both addSpecial=true); token IDs vs context array
// (server addSpecial=false, so our leading BOS is skipped before comparing).
func TestTokenizeGenerateMatchesAPI(t *testing.T) {
	ensureModelsDir(t)
	apiURL := ollamaURL(t)
	models := listModels(t)
	if len(models) == 0 {
		t.Fatalf("no models installed - set OLLAMA_MODELS and pull a model first")
	}

	for _, modelName := range models {
		modelName := modelName
		// Serial across models: the test server loads one model at a time, and
		// parallel load/unload races produce wrong API prompt_eval_counts.
		t.Run(modelName, func(t *testing.T) {
			tok, err := New(modelName)
			if err != nil {
				t.Fatalf("New(%q): %v", modelName, err)
			}

			for _, tm := range thinkModes {
				t.Run(tm.name, func(t *testing.T) {
					ourTokens, err := tok.TokenizeGenerate(api.GenerateRequest{
						Prompt: testPrompt(),
						Think:  tm.think,
					})
					if err != nil {
						if skipIfBothReject(t, err, func() ([]int, int, error) {
							return apiGenerate(apiURL, modelName, testPrompt(), tm.think)
						}) {
							return
						}
						t.Fatalf("TokenizeGenerate: %v", err)
					}

					apiTokens, promptEvalCount, err := apiGenerate(apiURL, modelName, testPrompt(), tm.think)
					if err != nil {
						if isUnsupportedError(err) {
							t.Skipf("unsupported: %v", err)
						}
						t.Fatalf("API /generate: %v", err)
					}

					// The live server reports transient wrong counts while
					// models load/unload under OLLAMA_MAX_LOADED_MODELS=1;
					// retry once before failing.
					if len(ourTokens) != promptEvalCount {
						time.Sleep(2 * time.Second)
						apiTokens, promptEvalCount, _ = apiGenerate(apiURL, modelName, testPrompt(), tm.think)
					}

					if len(ourTokens) != promptEvalCount {
						t.Errorf("token count mismatch: ours=%d API prompt_eval_count=%d",
							len(ourTokens), promptEvalCount)
					}

					ourPromptTokens := ourTokens
					if len(ourTokens) > 0 && len(apiTokens) > 0 && ourTokens[0] != int32(apiTokens[0]) {
						ourPromptTokens = ourTokens[1:]
					}

					if len(ourPromptTokens) > len(apiTokens) {
						t.Fatalf("our prompt tokens (%d) longer than API context (%d)", len(ourPromptTokens), len(apiTokens))
					}

					apiPromptTokens := apiTokens[:len(ourPromptTokens)]
					if !tokenSlicesEqual(ourPromptTokens, apiPromptTokens) {
						firstDiff := -1
						for i := range ourPromptTokens {
							if int(ourPromptTokens[i]) != apiTokens[i] {
								firstDiff = i
								break
							}
						}
						t.Errorf("token mismatch at position %d:\n  ours: %v\n  API:  %v\n  (our len=%d, API prompt_eval_count=%d, API context len=%d)",
							firstDiff, ourPromptTokens, apiPromptTokens, len(ourTokens), promptEvalCount, len(apiTokens))
					}
				})
			}
		})
	}
}

// TestTokenizeChatMatchesAPI compares our token count to live /api/chat.
func TestTokenizeChatMatchesAPI(t *testing.T) {
	ensureModelsDir(t)
	apiURL := ollamaURL(t)
	models := listModels(t)
	if len(models) == 0 {
		t.Skip("no models installed")
	}

	for _, modelName := range models {
		modelName := modelName
		// Serial across models: the test server loads one model at a time, and
		// parallel load/unload races produce wrong API prompt_eval_counts.
		t.Run(modelName, func(t *testing.T) {
			tok, err := New(modelName)
			if err != nil {
				t.Fatalf("New(%q): %v", modelName, err)
			}

			for _, tm := range thinkModes {
				t.Run(tm.name, func(t *testing.T) {
					ourTokens, err := tok.TokenizeChat(api.ChatRequest{
						Messages: testMessages,
						Think:    tm.think,
					})
					if err != nil {
						if skipIfBothReject(t, err, func() ([]int, int, error) {
							count, err := apiChat(apiURL, modelName, testMessages, tm.think)
							return nil, count, err
						}) {
							return
						}
						t.Fatalf("TokenizeChat: %v", err)
					}

					apiCount, err := apiChat(apiURL, modelName, testMessages, tm.think)
					if err != nil {
						if isUnsupportedError(err) {
							t.Skipf("unsupported: %v", err)
						}
						t.Fatalf("API /api/chat: %v", err)
					}

					// Retry once against transient server count races (see above).
					if len(ourTokens) != apiCount {
						time.Sleep(2 * time.Second)
						apiCount, _ = apiChat(apiURL, modelName, testMessages, tm.think)
					}

					if len(ourTokens) != apiCount {
						t.Errorf("token count mismatch: ours=%d API=%d",
							len(ourTokens), apiCount)
					}
				})
			}
		})
	}
}

// TestPrefillParity checks assistant-prefill (trailing assistant message):
// counts must match live /api/chat, and the invalid shapes both sides reject.
func TestPrefillParity(t *testing.T) {
	ensureModelsDir(t)
	apiURL := ollamaURL(t)
	models := listModels(t)
	if len(models) == 0 {
		t.Skip("no models installed")
	}

	prefillMsgs := []api.Message{
		{Role: "user", Content: "q1"},
		{Role: "assistant", Content: "partial answer"},
	}
	doubleAssistant := []api.Message{
		{Role: "user", Content: "q1"},
		{Role: "assistant", Content: "a1"},
		{Role: "assistant", Content: "a2"},
	}

	for _, modelName := range models {
		modelName := modelName
		// Serial across models: the test server loads one model at a time, and
		// parallel load/unload races produce wrong API prompt_eval_counts.
		t.Run(modelName, func(t *testing.T) {
			tok, err := New(modelName)
			if err != nil {
				t.Fatalf("New(%q): %v", modelName, err)
			}

			t.Run("prefill", func(t *testing.T) {
				ourTokens, err := tok.TokenizeChat(api.ChatRequest{Messages: prefillMsgs})
				if err != nil {
					t.Fatalf("TokenizeChat: %v", err)
				}
				apiCount, err := apiChat(apiURL, modelName, prefillMsgs, nil)
				if err != nil {
					t.Skipf("API rejected prefill: %v", err)
				}
				if len(ourTokens) != apiCount {
					t.Errorf("prefill count mismatch: ours=%d API=%d", len(ourTokens), apiCount)
				}
			})

			t.Run("double_assistant", func(t *testing.T) {
				_, err := tok.TokenizeChat(api.ChatRequest{Messages: doubleAssistant})
				if err == nil {
					t.Skip("model renders trailing assistants without continuation")
				}
				if _, apiErr := apiChat(apiURL, modelName, doubleAssistant, nil); apiErr == nil {
					t.Errorf("we reject but the API accepts: %v", err)
				}
			})
		})
	}
}

// --- API helpers ---

type apiError struct {
	statusCode int
	body       string
}

func (e *apiError) Error() string { return fmt.Sprintf("status %d: %s", e.statusCode, e.body) }

func isUnsupportedError(err error) bool {
	if apiErr, ok := err.(*apiError); ok {
		return apiErr.statusCode == 400 &&
			(strings.Contains(apiErr.body, "does not support thinking") ||
				strings.Contains(apiErr.body, "does not support generate") ||
				strings.Contains(apiErr.body, "does not support chat"))
	}
	return false
}

// skipIfBothReject skips when both we and the live API reject (err must be a
// BadRequestError); it fatals if the API accepts what we rejected.
func skipIfBothReject(t *testing.T, err error, callAPI func() ([]int, int, error)) bool {
	t.Helper()
	var badReq *BadRequestError
	if !errors.As(err, &badReq) {
		return false
	}
	if _, _, apiErr := callAPI(); apiErr != nil {
		t.Skipf("both reject: ours=%v api=%v", err, apiErr)
	}
	t.Fatalf("we reject but the API does not: %v", err)
	return true
}

func doAPIRequest(url string, body map[string]any) (*http.Response, error) {
	b, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}
	resp, err := http.Post(url, "application/json", bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, &apiError{statusCode: resp.StatusCode, body: string(b)}
	}
	return resp, nil
}

func apiRequestBase(model string, think *api.ThinkValue) map[string]any {
	req := map[string]any{
		"model":      model,
		"stream":     false,
		"keep_alive": 0,
		"options":    map[string]any{"num_predict": 0},
	}
	if think != nil {
		// Send the raw value (bool or string level), like a real client would.
		req["think"] = think.Value
	}
	return req
}

func apiGenerate(baseURL, model, prompt string, think *api.ThinkValue) ([]int, int, error) {
	req := apiRequestBase(model, think)
	req["prompt"] = prompt

	resp, err := doAPIRequest(baseURL+"/api/generate", req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()

	var result struct {
		Context         []int `json:"context"`
		PromptEvalCount int   `json:"prompt_eval_count"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, 0, fmt.Errorf("decode response: %w", err)
	}
	return result.Context, result.PromptEvalCount, nil
}

func apiChat(baseURL, model string, msgs []api.Message, think *api.ThinkValue) (int, error) {
	apiMsgs := make([]map[string]string, len(msgs))
	for i, m := range msgs {
		apiMsgs[i] = map[string]string{"role": m.Role, "content": m.Content}
	}

	req := apiRequestBase(model, think)
	req["messages"] = apiMsgs

	resp, err := doAPIRequest(baseURL+"/api/chat", req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()

	var result struct {
		PromptEvalCount int `json:"prompt_eval_count"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return 0, fmt.Errorf("decode response: %w", err)
	}
	return result.PromptEvalCount, nil
}

func tokenSlicesEqual(a []int32, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if int(a[i]) != b[i] {
			return false
		}
	}
	return true
}
