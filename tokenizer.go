// Package ollamatokenizer exposes ollama's tokenization without loading model
// weights, by linking ollama's bundled libllama.so and loading each GGUF
// vocab-only. Token IDs are byte-identical to a running ollama server.
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

// ErrNotImplemented is returned for unsupported request options.
var ErrNotImplemented = fmt.Errorf("not implemented")

// Tokenizer encodes text for a model using llama.cpp's real tokenizer, loaded vocab-only.
type Tokenizer struct {
	tok   *cgoVocab
	model *server.Model
}

// New returns a Tokenizer for a pulled model; only the GGUF vocab is read.
// https://github.com/ollama/ollama/blob/v0.30.11/server/images.go#L641
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

// Close releases the vocab handle.
func (t *Tokenizer) Close() {
	if t != nil && t.tok != nil {
		t.tok.Close()
	}
}

// IsNativeJinja reports whether ollama would render this model via llama-server's
// Jinja engine (minja), which we approximate in-process — see renderNativeJinja.
func (t *Tokenizer) IsNativeJinja() bool { return nativeJinja(t.model) }

// Tokenize encodes raw text. addSpecial applies BOS/EOS per the vocab; parseSpecial
// parses special-token strings (e.g. <|im_start|>) into their IDs.
func (t *Tokenizer) Tokenize(text string, addSpecial, parseSpecial bool) ([]int32, error) {
	tokens, err := t.tok.Encode(text, addSpecial, parseSpecial)
	if err != nil {
		return nil, fmt.Errorf(errPfx+"tokenize: %w", err)
	}
	return tokens, nil
}

// hasThinking mirrors server.routes thinking detection.
// https://github.com/ollama/ollama/blob/v0.30.11/server/routes.go#L2640-L2648
func (t *Tokenizer) hasThinking() bool {
	return slices.Contains(t.model.Capabilities(), modelname.CapabilityThinking)
}

// resolveThink defaults think to true for thinking-capable models when unset.
// https://github.com/ollama/ollama/blob/v0.30.11/server/routes.go#L2640-L2648
func (t *Tokenizer) resolveThink(think *api.ThinkValue) *api.ThinkValue {
	if think != nil {
		return think
	}
	if t.hasThinking() {
		return &api.ThinkValue{Value: true}
	}
	return nil
}

// renderPrompt mirrors server.renderPrompt, plus an in-process native-Jinja path.
// https://github.com/ollama/ollama/blob/v0.30.11/server/prompt.go#L135-L156
func (t *Tokenizer) renderPrompt(msgs []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	if t.model.Config.Renderer != "" {
		rendered, err := renderers.RenderWithRenderer(resolveRendererName(t.model), msgs, tools, think)
		if err != nil {
			return "", fmt.Errorf(errPfx+"renderer %q: %w", t.model.Config.Renderer, err)
		}
		return rendered, nil
	}

	if nativeJinja(t.model) {
		return t.renderNativeJinja(msgs)
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
	if err := t.model.Template.Execute(&b, template.Values{
		Messages:   msgs,
		Tools:      tools,
		Think:      thinkVal,
		ThinkLevel: thinkLevel,
		IsThinkSet: think != nil,
	}); err != nil {
		return "", fmt.Errorf(errPfx+"template: %w", err)
	}
	return b.String(), nil
}

// nativeJinja mirrors chatModeForModel(m) == native for the no-Renderer/Parser case;
// PreferChatTemplate is set by ollama when the GGUF chat_template beats the Go TEMPLATE.
// https://github.com/ollama/ollama/blob/v0.30.11/server/routes.go#L2381-L2398
func nativeJinja(m *server.Model) bool {
	if m == nil || !m.HasChatTemplate {
		return false
	}
	if m.Config.Renderer != "" || m.Config.Parser != "" || shouldUseHarmony(m) {
		return false
	}
	return m.PreferChatTemplate || !m.HasGoTemplate
}

// renderNativeJinja approximates llama-server's Jinja (minja) — which lives only
// in the llama-server binary — using llama.cpp's builtin template engine plus
// two behaviors the builtin drops: the literal prefix baked before the message
// loop (e.g. phi4's embedded system), and merging system into the first user
// message when the loop has no system branch.
func (t *Tokenizer) renderNativeJinja(msgs []api.Message) (string, error) {
	tmpl := t.tok.ChatTemplate()
	loop := loopBody(tmpl)
	chatMsgs := msgs
	if loop != "" && !systemBranchInLoop(loop) {
		chatMsgs = mergeSystemIntoUser(chatMsgs)
	}
	cm := make([]ChatMessage, 0, len(chatMsgs))
	for _, m := range chatMsgs {
		cm = append(cm, ChatMessage{Role: m.Role, Content: m.Content})
	}
	rendered, err := t.tok.RenderChat(cm, true)
	if err != nil {
		return "", fmt.Errorf(errPfx+"native chat template for %q: %w", t.model.Name, err)
	}
	if baked := bakedLiteralPrefix(tmpl); baked != "" {
		rendered = baked + rendered
	}
	return rendered, nil
}

// bakedLiteralPrefix returns literal text the template emits before {% for },
// or "" if that region holds Jinja logic rather than verbatim text.
func bakedLiteralPrefix(tmpl string) string {
	i := strings.Index(tmpl, "{% for")
	if i < 0 {
		i = strings.Index(tmpl, "{%- for")
	}
	if i <= 0 {
		return ""
	}
	head := tmpl[:i]
	if strings.Contains(head, "{%") || strings.Contains(head, "{{") {
		return ""
	}
	return head
}

// loopBody returns the {% for %}...{% endfor %} substring, or "" if absent.
func loopBody(tmpl string) string {
	i := strings.Index(tmpl, "{% for")
	if i < 0 {
		i = strings.Index(tmpl, "{%- for")
	}
	if i < 0 {
		return ""
	}
	if j := strings.Index(tmpl[i:], "{% endfor"); j >= 0 {
		return tmpl[i : i+j]
	}
	return ""
}

// systemBranchInLoop reports whether the loop handles role == "system".
func systemBranchInLoop(loop string) bool { return strings.Contains(loop, "system") }

// mergeSystemIntoUser prepends all system contents to the first user message,
// matching llama.cpp's handling of templates whose loop ignores system.
func mergeSystemIntoUser(msgs []api.Message) []api.Message {
	var sys []string
	rest := msgs[:0:0]
	for _, m := range msgs {
		if m.Role == "system" {
			sys = append(sys, m.Content)
		} else {
			rest = append(rest, m)
		}
	}
	if len(sys) == 0 {
		return msgs
	}
	merged := strings.Join(sys, "\n")
	for i, m := range rest {
		if m.Role == "user" {
			rest[i].Content = merged + "\n" + m.Content
			return rest
		}
	}
	return append([]api.Message{{Role: "user", Content: merged}}, rest...)
}

// completionPrompt mirrors llamaServerRunner.completionPrompt: strip the textual
// BOS the renderer emitted when llama.cpp will also add BOS, else it's counted twice.
// https://github.com/ollama/ollama/blob/v0.30.11/llm/llama_server.go#L232-L244
func (t *Tokenizer) completionPrompt(prompt string) string {
	if !t.tok.AddBOS() {
		return prompt
	}
	if t.model.Config.Renderer != "" {
		if lb := renderers.LeadingBOSForRenderer(resolveRendererName(t.model)); lb != "" && strings.HasPrefix(prompt, lb) {
			return strings.TrimPrefix(prompt, lb)
		}
	}
	if strings.HasPrefix(prompt, "<bos>") {
		return strings.TrimPrefix(prompt, "<bos>")
	}
	return prompt
}

// filterThinkTags strips <think> from prior assistant turns for qwen3 / deepseek-r1.
// https://github.com/ollama/ollama/blob/v0.30.11/server/routes.go#L3141-L3167
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

// shouldUseHarmony mirrors server.shouldUseHarmony (gpt-oss).
// https://github.com/ollama/ollama/blob/v0.30.11/server/routes.go#L80-L90
func shouldUseHarmony(m *server.Model) bool {
	if slices.Contains([]string{"gptoss", "gpt-oss"}, m.Config.ModelFamily) {
		if m.Template.Contains("<|start|>") && m.Template.Contains("<|end|>") {
			return true
		}
	}
	return false
}

// processTools mirrors the chat handler's harmony + builtin-parser setup.
// https://github.com/ollama/ollama/blob/v0.30.11/server/routes.go#L2686-L2718
func processTools(m *server.Model, tools []api.Tool, msgs []api.Message, think *api.ThinkValue) []api.Tool {
	if shouldUseHarmony(m) {
		// harmony only understands low/medium/high; map "max" -> "high".
		if think != nil {
			if s, ok := think.Value.(string); ok && s == "max" {
				think.Value = "high"
			}
		}
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

// TokenizeGenerate mirrors /api/generate's prompt assembly (no context truncation).
// Unsupported (ErrNotImplemented): Suffix, Template, Raw, Context, Images.
// https://github.com/ollama/ollama/blob/v0.30.11/server/routes.go#L528-L542
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

	var msgs []api.Message
	if req.System != "" {
		msgs = append(msgs, api.Message{Role: "system", Content: req.System})
	} else if t.model.System != "" {
		msgs = append(msgs, api.Message{Role: "system", Content: t.model.System})
	}
	msgs = append(msgs, t.model.Messages...)
	msgs = append(msgs, api.Message{Role: "user", Content: req.Prompt})

	rendered, err := t.renderPrompt(msgs, nil, t.resolveThink(req.Think))
	if err != nil {
		return nil, err
	}
	return t.Tokenize(t.completionPrompt(rendered), true, true)
}

// TokenizeChat mirrors /api/chat's prompt assembly (no context truncation).
// https://github.com/ollama/ollama/blob/v0.30.11/server/routes.go#L2680-L2724
func (t *Tokenizer) TokenizeChat(req api.ChatRequest) ([]int32, error) {
	msgs := append(t.model.Messages, req.Messages...)
	if len(req.Messages) > 0 && req.Messages[0].Role != "system" && t.model.System != "" {
		msgs = append([]api.Message{{Role: "system", Content: t.model.System}}, msgs...)
	}
	msgs = filterThinkTags(msgs, t.model)

	think := t.resolveThink(req.Think)
	processedTools := processTools(t.model, req.Tools, msgs, think)

	rendered, err := t.renderPrompt(msgs, processedTools, think)
	if err != nil {
		return nil, err
	}
	return t.Tokenize(t.completionPrompt(rendered), true, true)
}

// gemma4 renderer resolution — verbatim mirror of server/renderer_resolution.go.
// https://github.com/ollama/ollama/blob/v0.30.11/server/renderer_resolution.go

func resolveRendererName(m *server.Model) string {
	if m == nil || m.Config.Renderer == "" {
		return ""
	}
	if m.Config.Renderer == "gemma4" {
		return resolveGemma4Renderer(m)
	}
	return m.Config.Renderer
}

func resolveGemma4Renderer(m *server.Model) string {
	if m == nil || m.Config.Renderer != "gemma4" {
		if m == nil {
			return "gemma4"
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
	return "gemma4-small"
}

const gemma4LargeMinParameterCount = 12_000_000_000

func gemma4RendererForParameterCount(parameterCount uint64) string {
	if parameterCount >= gemma4LargeMinParameterCount {
		return "gemma4-large"
	}
	return "gemma4-small"
}

func gemma4RendererFromName(name string) (string, bool) {
	lower := strings.ToLower(name)
	switch {
	case strings.Contains(lower, "e2b"), strings.Contains(lower, "e4b"):
		return "gemma4-small", true
	case strings.Contains(lower, "12b"), strings.Contains(lower, "26b"), strings.Contains(lower, "31b"):
		return "gemma4-large", true
	default:
		return "", false
	}
}

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
