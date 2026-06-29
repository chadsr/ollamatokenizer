// Package ollamatokenizer exposes ollama's tokenization without loading model
// weights. A tokenizer is built from the model's GGUF vocabulary via ollama's
// pure-Go tokenizer package; chat prompts are rendered with ollama's template,
// renderer, and parser packages. No weights, GPU, or subprocess — token IDs
// match ollama for any BPE or SentencePiece model (essentially all of them).
package ollamatokenizer

import (
	"bytes"
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	fsggml "github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/model/parsers"
	"github.com/ollama/ollama/model/renderers"
	"github.com/ollama/ollama/server"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/thinking"
	"github.com/ollama/ollama/tokenizer"
	modelname "github.com/ollama/ollama/types/model"
)

const errPfx = "ollamatokenizer: "

// ErrNotImplemented is returned for request options this library does not support.
var ErrNotImplemented = fmt.Errorf("not implemented")

// Tokenizer encodes text for a model using ollama's pure-Go tokenizer built from
// the model's GGUF vocabulary. Only the vocabulary is held in memory.
type Tokenizer struct {
	tok   tokenizer.Tokenizer
	model *server.Model
}

// New returns a Tokenizer for the named model (e.g. "llama3.2:3b"); the model
// must be pulled. Only the GGUF vocabulary is read — no weights are loaded.
func New(name string) (*Tokenizer, error) {
	// https://github.com/ollama/ollama/blob/v0.30.11/server/images.go#L641
	m, err := server.GetModel(name)
	if err != nil {
		return nil, fmt.Errorf(errPfx+"model %q not found (try `ollama pull %s`): %w", name, name, err)
	}

	f, err := os.Open(m.ModelPath)
	if err != nil {
		return nil, fmt.Errorf(errPfx+"open model file: %w", err)
	}
	defer f.Close()

	ggmlFile, err := fsggml.Decode(f, -1)
	if err != nil {
		return nil, fmt.Errorf(errPfx+"decode GGUF metadata: %w", err)
	}

	tok, err := newTokenizer(ggmlFile.KV())
	if err != nil {
		return nil, fmt.Errorf(errPfx+"model %q: %w", name, err)
	}

	return &Tokenizer{tok: tok, model: m}, nil
}

// newTokenizer builds a tokenizer from GGUF vocabulary fields (the same fields
// every model carries), so it is architecture-independent: "gpt2" → BPE,
// "llama" → SentencePiece.
func newTokenizer(kv fsggml.KV) (tokenizer.Tokenizer, error) {
	vocab := &tokenizer.Vocabulary{
		Values: kv.Strings("tokenizer.ggml.tokens"),
		Scores: kv.Floats("tokenizer.ggml.scores"),
		Types:  kv.Ints("tokenizer.ggml.token_type"),
		Merges: kv.Strings("tokenizer.ggml.merges"),
		AddBOS: kv.Bool("tokenizer.ggml.add_bos_token", false),
		BOS:    []int32{int32(kv.Uint("tokenizer.ggml.bos_token_id"))},
		AddEOS: kv.Bool("tokenizer.ggml.add_eos_token", false),
		EOS: append(
			[]int32{int32(kv.Uint("tokenizer.ggml.eos_token_id"))},
			kv.Ints("tokenizer.ggml.eos_token_ids")...,
		),
	}

	switch kv.String("tokenizer.ggml.model") {
	case "gpt2":
		// BPE pre-split regex, keyed by the GGUF `tokenizer.ggml.pre` string.
		// Source of truth: llama.cpp's pre-tokenizer table.
		// https://github.com/ggml-org/llama.cpp/blob/master/src/llama-vocab.cpp
		var pretokenizers []string
		switch kv.String("tokenizer.ggml.pre") {
		case "llama-bpe", "llama3", "llama-v3", "dbrx", "smaug", "chatglm-bpe",
			"falcon3", "falcon-h1", "pixtral", "midm-2.0", "lfm2", "jina-v5-nano":
			pretokenizers = []string{
				"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
			}
		case "qwen2", "deepseek-r1-qwen":
			pretokenizers = []string{
				"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
			}
		case "qwen35":
			pretokenizers = []string{
				"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
			}
		case "refact":
			pretokenizers = []string{
				`\p{N}`,
				`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`,
			}
		}
		// Unlisted ("default", "gpt-2", ...) → built-in GPT-2 byte-level pretokenizer.
		// https://github.com/ollama/ollama/blob/v0.30.11/tokenizer/bytepairencoding.go#L35
		return tokenizer.NewBytePairEncoding(vocab, pretokenizers...), nil
	case "llama":
		// https://github.com/ollama/ollama/blob/v0.30.11/tokenizer/sentencepiece.go#L26
		return tokenizer.NewSentencePiece(vocab), nil
	default:
		return nil, fmt.Errorf("unsupported tokenizer model %q (only gpt2/llama are supported)", kv.String("tokenizer.ggml.model"))
	}
}

// Tokenize encodes raw text into token IDs. addSpecial prepends BOS / appends
// EOS per the model's vocabulary. Special-token strings (e.g. <|im_start|>) are
// always parsed.
func (t *Tokenizer) Tokenize(text string, addSpecial, _ bool) ([]int32, error) {
	tokens, err := t.tok.Encode(text, addSpecial)
	if err != nil {
		return nil, fmt.Errorf(errPfx+"tokenize: %w", err)
	}
	return tokens, nil
}

// --- Prompt rendering: mirrors of unexported helpers in ollama's server package,
// kept in sync with the pinned version (see go.mod). ---

// hasThinking reports whether the model supports thinking.
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

// renderPrompt renders messages via the model's Go renderer or Modelfile
// TEMPLATE (ollama's path for any model with a Go template).
// https://github.com/ollama/ollama/blob/v0.30.11/server/prompt.go#L135-L156
func (t *Tokenizer) renderPrompt(msgs []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	if t.model.Config.Renderer != "" {
		rendered, err := renderers.RenderWithRenderer(resolveRendererName(t.model), msgs, tools, think)
		if err != nil {
			return "", fmt.Errorf(errPfx+"renderer %q: %w", t.model.Config.Renderer, err)
		}
		return rendered, nil
	}

	if t.model.Template == nil {
		return "", fmt.Errorf(errPfx+"model %q has no Go chat template (native-Jinja models need the runner): %w", t.model.Name, ErrNotImplemented)
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

// filterThinkTags strips <think> content from assistant messages preceding the
// final user message, for qwen3 and deepseek-r1.
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

// shouldUseHarmony detects harmony-based models (gpt-oss).
// https://github.com/ollama/ollama/blob/v0.30.11/server/routes.go#L80-L90
func shouldUseHarmony(m *server.Model) bool {
	if slices.Contains([]string{"gptoss", "gpt-oss"}, m.Config.ModelFamily) {
		if m.Template.Contains("<|start|>") && m.Template.Contains("<|end|>") {
			return true
		}
	}
	return false
}

// processTools runs the chat handler's harmony + builtin-parser setup.
// https://github.com/ollama/ollama/blob/v0.30.11/server/routes.go#L2686-L2718
func processTools(m *server.Model, tools []api.Tool, msgs []api.Message, think *api.ThinkValue) []api.Tool {
	if shouldUseHarmony(m) {
		// harmony only understands low/medium/high; map "max" → "high".
		// https://github.com/ollama/ollama/blob/v0.30.11/server/routes.go#L2686-L2694
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

// TokenizeGenerate tokenizes a prompt matching /api/generate. Context-length
// truncation is not replicated (matches for prompts within the window).
// Unsupported (returns ErrNotImplemented): Suffix, Template, Raw, Context, Images.
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
	// addSpecial matches the runner's prompt prefill counting (BOS/EOS per vocab).
	return t.Tokenize(rendered, true, true)
}

// TokenizeChat tokenizes messages matching /api/chat. Context-length truncation
// is not replicated (matches for prompts within the window).
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
	return t.Tokenize(rendered, true, true)
}

// --- gemma4 renderer resolution (mirror of server/renderer_resolution.go) ---
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

// gemma4LargeMinParameterCount is the threshold above which gemma4 uses the large renderer.
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
