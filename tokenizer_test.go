package ollamatokenizer

import (
	"bytes"
	"encoding/json"
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

var thinkModes = []struct {
	name  string
	think *api.ThinkValue
}{
	{"think_nil", nil},
	{"think_true", &api.ThinkValue{Value: true}},
	{"think_false", &api.ThinkValue{Value: false}},
}

// edgeCasePrompt exercises Unicode (CJK, emoji, accented), whitespace (tabs,
// newlines, multiple spaces), punctuation/symbols, code syntax, and URLs.
const edgeCasePrompt = "Hello, 世界! \tWhat is 2+2?\nLine two.   Extra  spaces.\n" +
	"café résumé naïve — \"quotes\" 'apos' <html> &amp;\n" +
	"emoji: 🌍🤖✅\n" +
	"code: func main() { fmt.Println(\"hi\") }\n" +
	"$100 @user #tag https://example.com/path?q=1&r=2"

// TestTokenizeGenerateMatchesAPI verifies TokenizeGenerate produces tokens
// identical to the Ollama /api/generate endpoint.
func TestTokenizeGenerateMatchesAPI(t *testing.T) {
	ensureModelsDir(t)
	apiURL := ollamaURL(t)
	models := listModels(t)
	if len(models) == 0 {
		t.Skip("no models installed — set OLLAMA_MODELS and pull a model first")
	}

	for _, modelName := range models {
		for _, tm := range thinkModes {
			t.Run(fmt.Sprintf("%s/%s", modelName, tm.name), func(t *testing.T) {
				t.Parallel()

				tok, err := New(modelName)
				if err != nil {
					t.Fatalf("New(%q): %v", modelName, err)
				}

				ourTokens, err := tok.TokenizeGenerate(api.GenerateRequest{
					Prompt: edgeCasePrompt,
					Think:  tm.think,
				})
				if err != nil {
					t.Fatalf("TokenizeGenerate: %v", err)
				}

				apiTokens, promptEvalCount, err := apiGenerate(apiURL, modelName, edgeCasePrompt, tm.think)
				if err != nil {
					if isUnsupportedError(err) {
						t.Skipf("unsupported: %v", err)
					}
					t.Fatalf("API /generate: %v", err)
				}

				// API's prompt_eval_count may include 1 extra prefill token, so compare our prefix.
				if len(ourTokens) > len(apiTokens) {
					t.Fatalf("our tokens (%d) longer than API context (%d)", len(ourTokens), len(apiTokens))
				}

				apiPromptTokens := apiTokens[:len(ourTokens)]
				if !tokenSlicesEqual(ourTokens, apiPromptTokens) {
					firstDiff := -1
					for i := range ourTokens {
						if int(ourTokens[i]) != apiTokens[i] {
							firstDiff = i
							break
						}
					}
					t.Errorf("token mismatch at position %d:\n  ours: %v\n  API:  %v\n  (our len=%d, API prompt_eval_count=%d, API total=%d)",
						firstDiff, ourTokens, apiPromptTokens, len(ourTokens), promptEvalCount, len(apiTokens))
				}
			})
		}
	}
}

// TestTokenizeChatMatchesAPI verifies TokenizeChat produces the same token
// count as the Ollama /api/chat endpoint (which doesn't return a context array).
func TestTokenizeChatMatchesAPI(t *testing.T) {
	ensureModelsDir(t)
	apiURL := ollamaURL(t)
	models := listModels(t)
	if len(models) == 0 {
		t.Skip("no models installed")
	}

	chatMessages := []api.Message{
		{Role: "system", Content: "You are a helpful assistant who speaks concisely."},
		{Role: "user", Content: "What is the capital of France?"},
		{Role: "assistant", Content: "The capital of France is Paris."},
		{Role: "user", Content: "And Germany?"},
	}

	for _, modelName := range models {
		for _, tm := range thinkModes {
			t.Run(fmt.Sprintf("%s/%s", modelName, tm.name), func(t *testing.T) {
				t.Parallel()

				tok, err := New(modelName)
				if err != nil {
					t.Fatalf("New(%q): %v", modelName, err)
				}

				ourTokens, err := tok.TokenizeChat(api.ChatRequest{
					Messages: chatMessages,
					Think:    tm.think,
				})
				if err != nil {
					t.Fatalf("TokenizeChat: %v", err)
				}

				apiCount, err := apiChat(apiURL, modelName, chatMessages, tm.think)
				if err != nil {
					if isUnsupportedError(err) {
						t.Skipf("unsupported: %v", err)
					}
					t.Fatalf("API /api/chat: %v", err)
				}

				// API's prompt_eval_count may include 1 extra prefill token.
				diff := apiCount - len(ourTokens)
				if diff < 0 || diff > 1 {
					t.Errorf("token count mismatch: ours=%d API=%d (diff=%d)",
						len(ourTokens), apiCount, diff)
				}
			})
		}
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
				strings.Contains(apiErr.body, "does not support generate"))
	}
	return false
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
		req["think"] = think.Bool()
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
