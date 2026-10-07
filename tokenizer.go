// Package ollamatokenizer: token IDs identical to a running ollama server,
// without loading model weights (vocab-only).
package ollamatokenizer

import (
	"bytes"
	"fmt"
	"slices"
	"strconv"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/model/parsers"
	"github.com/ollama/ollama/model/renderers"
	"github.com/ollama/ollama/server"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/thinking"
	modelname "github.com/ollama/ollama/types/model"
)

const errPfx = "ollamatokenizer: "

var ErrNotImplemented = fmt.Errorf("not implemented")

// BadRequestError marks errors ollama answers with HTTP 400.
type BadRequestError struct{ Err error }

func (e *BadRequestError) Error() string { return e.Err.Error() }
func (e *BadRequestError) Unwrap() error { return e.Err }

// Tokenizer holds a vocab-only GGUF handle and ollama model metadata.
type Tokenizer struct {
	tok   *cgoVocab
	model *server.Model
}

// New loads a model's vocab-only GGUF via cgo and ollama metadata via server.GetModel.
// https://github.com/ollama/ollama/blob/v0.40.0/server/images.go#L706
func New(name string) (*Tokenizer, error) {
	m, err := server.GetModel(name)
	if err != nil {
		return nil, fmt.Errorf(errPfx+"model %q not found (try `ollama pull %s`): %w", name, name, err)
	}
	tok, err := newCGOVocab(m.ModelPath)
	if err != nil {
		return nil, fmt.Errorf(errPfx+"model %q: %w", name, err)
	}
	return &Tokenizer{tok: tok, model: m}, nil
}

func (t *Tokenizer) Close() {
	if t != nil && t.tok != nil {
		t.tok.Close()
	}
}

// Tokenize encodes text; parseSpecial parses special-token strings.
func (t *Tokenizer) Tokenize(text string, addSpecial, parseSpecial bool) ([]int32, error) {
	tokens, err := t.tok.Encode(text, addSpecial, parseSpecial)
	if err != nil {
		return nil, fmt.Errorf(errPfx+"tokenize: %w", err)
	}
	return tokens, nil
}

// genericThinking mirrors Model.genericThinking (renderer descriptors only).
// https://github.com/ollama/ollama/blob/v0.40.0/server/model_thinking.go#L98-L103
func genericThinking(m *server.Model) *modelname.Thinking {
	if m == nil || m.Config.Renderer == "" || shouldUseHarmony(m) {
		return nil
	}
	return m.Thinking()
}

// resolveThink mirrors the handlers' think resolution.
// https://github.com/ollama/ollama/blob/v0.40.0/server/routes.go#L509-L536
// https://github.com/ollama/ollama/blob/v0.40.0/server/routes.go#L3087-L3125
func (t *Tokenizer) resolveThink(reqThink *api.ThinkValue) (*api.ThinkValue, error) {
	thinking := genericThinking(t.model)
	if thinking == nil {
		if err := api.ValidateLegacyThinking(reqThink); err != nil {
			return nil, &BadRequestError{Err: err}
		}
	}

	think := renderers.ResolveThinking(reqThink, thinking)
	if slices.Contains(t.model.Capabilities(), modelname.CapabilityThinking) {
		if think == nil && thinking == nil {
			think = &api.ThinkValue{Value: true}
		}
	} else if think != nil && think.Bool() {
		return nil, &BadRequestError{Err: fmt.Errorf("%q does not support thinking", t.model.Name)}
	}
	return think, nil
}

// renderPrompt mirrors server.renderPrompt, with native Jinja dispatched inside.
// https://github.com/ollama/ollama/blob/v0.40.0/server/prompt.go#L135-L155
func (t *Tokenizer) renderPrompt(msgs []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	if t.model.Config.Renderer != "" {
		rendered, err := renderers.RenderWithRenderer(resolveRendererName(t.model), msgs, tools, think)
		if err != nil {
			return "", fmt.Errorf(errPfx+"renderer %q: %w", t.model.Config.Renderer, err)
		}
		return rendered, nil
	}

	if nativeJinja(t.model) {
		return t.renderNativeJinja(msgs, nil, think)
	}

	if t.model.Template == nil {
		return "", fmt.Errorf(errPfx+"model %q has no Go chat template: %w", t.model.Name, ErrNotImplemented)
	}

	var b bytes.Buffer
	thinkVal := false
	thinkLevel := ""
	if think != nil {
		thinkVal = think.Bool()
		thinkLevel = think.String()
	}
	if err := t.model.Template.Execute(&b, template.Values{Messages: msgs, Tools: tools, Think: thinkVal, ThinkLevel: thinkLevel, IsThinkSet: think != nil}); err != nil {
		return "", fmt.Errorf(errPfx+"template: %w", err)
	}
	return b.String(), nil
}

// nativeJinja is chatModeForModel(m)==native, collapsed.
// https://github.com/ollama/ollama/blob/v0.40.0/server/routes.go#L2809-L2838
func nativeJinja(m *server.Model) bool {
	if m == nil || !m.HasChatTemplate {
		return false
	}
	if m.Config.Renderer != "" || m.Config.Parser != "" || shouldUseHarmony(m) {
		return false
	}
	return m.PreferChatTemplate || !m.HasGoTemplate
}

// renderNativeJinja applies the GGUF chat template (see RenderChatJinja).
// https://github.com/ollama/ollama/blob/v0.40.0/llm/llama_server.go#L1945-L1986
func (t *Tokenizer) renderNativeJinja(msgs []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	rendered, err := t.tok.RenderChatJinja(msgs, tools, think)
	if err != nil {
		return "", fmt.Errorf(errPfx+"native chat template for %q: %w", t.model.Name, err)
	}
	return rendered, nil
}

// completionPrompt strips a textual BOS when the tokenizer adds BOS itself.
// https://github.com/ollama/ollama/blob/v0.40.0/llm/llama_server.go#L246-L258
func (t *Tokenizer) completionPrompt(prompt string) string {
	if t.tok.AddBOS() {
		if leadingBOS := renderers.LeadingBOSForRenderer(resolveRendererName(t.model)); leadingBOS != "" && strings.HasPrefix(prompt, leadingBOS) {
			return strings.TrimPrefix(prompt, leadingBOS)
		}
		if strings.HasPrefix(prompt, "<bos>") {
			return strings.TrimPrefix(prompt, "<bos>")
		}
	}
	return prompt
}

// filterThinkTags strips <think> from prior assistant turns for qwen3 / deepseek-r1.
// https://github.com/ollama/ollama/blob/v0.40.0/server/routes.go#L3537-L3563
func filterThinkTags(msgs []api.Message, m *server.Model) []api.Message {
	if m.Config.ModelFamily == "qwen3" || modelname.ParseName(m.Name).Model == "deepseek-r1" {
		finalUserIndex := -1
		for i, msg := range msgs {
			if msg.Role == "user" {
				finalUserIndex = i
			}
		}
		for i, msg := range msgs {
			if msg.Role == "assistant" && i < finalUserIndex {
				thinkingState := &thinking.Parser{
					OpeningTag: "<think>",
					ClosingTag: "</think>",
				}
				_, content := thinkingState.AddContent(msg.Content)
				msgs[i].Content = content
			}
		}
	}
	return msgs
}

// shouldUseHarmony mirrors server.shouldUseHarmony.
// https://github.com/ollama/ollama/blob/v0.40.0/server/routes.go#L79-L91
func shouldUseHarmony(m *server.Model) bool {
	if slices.Contains([]string{"gptoss", "gpt-oss"}, m.Config.ModelFamily) {
		if m.Template.Contains("<|start|>") && m.Template.Contains("<|end|>") {
			return true
		}
	}
	return false
}

// mapHarmonyThink maps "max" to "high" (harmony has no max level).
// https://github.com/ollama/ollama/blob/v0.40.0/server/routes.go#L539-L547
func mapHarmonyThink(think *api.ThinkValue) {
	if think == nil {
		return
	}
	if s, ok := think.Value.(string); ok && s == "max" {
		think.Value = "high"
	}
}

// processTools mirrors the chat handler's harmony + parser setup.
// https://github.com/ollama/ollama/blob/v0.40.0/server/routes.go#L3150-L3181
func processTools(m *server.Model, tools []api.Tool, msgs []api.Message, think *api.ThinkValue) []api.Tool {
	if shouldUseHarmony(m) {
		mapHarmonyThink(think)
		if m.Config.Parser == "" {
			m.Config.Parser = "harmony"
		}
	}

	processedTools := tools
	if m.Config.Parser != "" {
		if p := parsers.ParserForName(m.Config.Parser); p != nil {
			var lastMessage *api.Message
			if len(msgs) > 0 {
				lastMessage = &msgs[len(msgs)-1]
			}
			processedTools = p.Init(tools, lastMessage, think)
		}
	}
	return processedTools
}

// TokenizeGenerate mirrors /api/generate's prompt assembly (no truncation).
// Unsupported: Suffix, Template, Raw, Context, Images.
// https://github.com/ollama/ollama/blob/v0.40.0/server/routes.go#L509-L686
func (t *Tokenizer) TokenizeGenerate(req api.GenerateRequest) ([]int32, error) {
	if req.Suffix != "" {
		return nil, fmt.Errorf(errPfx+"suffix (insert mode) is not implemented: %w", ErrNotImplemented)
	}
	if req.Template != "" {
		return nil, fmt.Errorf(errPfx+"template override is not implemented: %w", ErrNotImplemented)
	}
	if req.Raw {
		return nil, fmt.Errorf(errPfx+"raw mode is not implemented: %w", ErrNotImplemented)
	}
	if len(req.Context) > 0 {
		return nil, fmt.Errorf(errPfx+"context (deprecated) is not implemented: %w", ErrNotImplemented)
	}
	if len(req.Images) > 0 {
		return nil, fmt.Errorf(errPfx+"images (multimodal) is not implemented: %w", ErrNotImplemented)
	}

	think, err := t.resolveThink(req.Think)
	if err != nil {
		return nil, err
	}
	// Generate passes no tools, so the parser cannot affect the prompt
	// (routes.go#L539-L557); only the harmony think mapping applies.
	if shouldUseHarmony(t.model) {
		mapHarmonyThink(think)
	}

	var msgs []api.Message
	if req.System != "" {
		msgs = append(msgs, api.Message{Role: "system", Content: req.System})
	} else if t.model.System != "" {
		msgs = append(msgs, api.Message{Role: "system", Content: t.model.System})
	}
	msgs = append(msgs, t.model.Messages...)
	msgs = append(msgs, api.Message{Role: "user", Content: req.Prompt})

	rendered, err := t.renderPrompt(msgs, nil, think)
	if err != nil {
		return nil, err
	}
	return t.Tokenize(t.completionPrompt(rendered), true, true)
}

// TokenizeChat mirrors /api/chat's prompt assembly (no truncation).
// https://github.com/ollama/ollama/blob/v0.40.0/server/routes.go#L3087-L3193
func (t *Tokenizer) TokenizeChat(req api.ChatRequest) ([]int32, error) {
	msgs := append(t.model.Messages, req.Messages...)
	if len(req.Messages) > 0 && req.Messages[0].Role != "system" && t.model.System != "" {
		msgs = append([]api.Message{{Role: "system", Content: t.model.System}}, msgs...)
	}
	msgs = filterThinkTags(msgs, t.model)

	think, err := t.resolveThink(req.Think)
	if err != nil {
		return nil, err
	}

	// The native path gets raw tools, not parser-processed ones.
	if nativeJinja(t.model) {
		rendered, err := t.renderNativeJinja(msgs, req.Tools, think)
		if err != nil {
			return nil, err
		}
		return t.Tokenize(t.completionPrompt(rendered), true, true)
	}

	processedTools := processTools(t.model, req.Tools, msgs, think)
	rendered, err := t.renderPrompt(msgs, processedTools, think)
	if err != nil {
		return nil, err
	}
	return t.Tokenize(t.completionPrompt(rendered), true, true)
}

// Gemma4 renderer resolution
// https://github.com/ollama/ollama/blob/v0.40.0/server/renderer_resolution.go#L21-L91
const (
	gemma4RendererLegacy         = "gemma4"
	gemma4RendererSmall          = "gemma4-small"
	gemma4RendererLarge          = "gemma4-large"
	gemma4LargeMinParameterCount = 12_000_000_000
)

// https://github.com/ollama/ollama/blob/v0.40.0/server/renderer_resolution.go#L21-L32
func resolveRendererName(m *server.Model) string {
	if m == nil || m.Config.Renderer == "" {
		return ""
	}
	switch m.Config.Renderer {
	case gemma4RendererLegacy:
		return resolveGemma4Renderer(m)
	default:
		return m.Config.Renderer
	}
}

// https://github.com/ollama/ollama/blob/v0.40.0/server/renderer_resolution.go#L34-L55
func resolveGemma4Renderer(m *server.Model) string {
	if m == nil || m.Config.Renderer != gemma4RendererLegacy {
		if m == nil {
			return gemma4RendererLegacy
		}
		return m.Config.Renderer
	}
	if renderer, ok := gemma4RendererFromName(m.ShortName); ok {
		return renderer
	}
	if renderer, ok := gemma4RendererFromName(m.Name); ok {
		return renderer
	}
	if parameterCount, ok := parseHumanParameterCount(m.Config.ModelType); ok {
		return gemma4RendererForParameterCount(parameterCount)
	}
	return gemma4RendererSmall
}

// https://github.com/ollama/ollama/blob/v0.40.0/server/renderer_resolution.go#L57-L63
func gemma4RendererForParameterCount(parameterCount uint64) string {
	if parameterCount >= gemma4LargeMinParameterCount {
		return gemma4RendererLarge
	}
	return gemma4RendererSmall
}

// https://github.com/ollama/ollama/blob/v0.40.0/server/renderer_resolution.go#L65-L75
func gemma4RendererFromName(name string) (string, bool) {
	lower := strings.ToLower(name)
	switch {
	case strings.Contains(lower, "e2b"), strings.Contains(lower, "e4b"):
		return gemma4RendererSmall, true
	case strings.Contains(lower, "12b"), strings.Contains(lower, "26b"), strings.Contains(lower, "31b"):
		return gemma4RendererLarge, true
	default:
		return "", false
	}
}

// https://github.com/ollama/ollama/blob/v0.40.0/server/renderer_resolution.go#L77-L91
func parseHumanParameterCount(s string) (uint64, bool) {
	if s == "" {
		return 0, false
	}
	unit := strings.ToUpper(s[len(s)-1:])
	var multiplier float64
	switch unit {
	case "B":
		multiplier = float64(format.Billion)
	case "M":
		multiplier = float64(format.Million)
	case "K":
		multiplier = float64(format.Thousand)
	default:
		return 0, false
	}
	value, err := strconv.ParseFloat(s[:len(s)-1], 64)
	if err != nil {
		return 0, false
	}
	return uint64(value * multiplier), true
}
