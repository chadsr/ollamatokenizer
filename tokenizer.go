// Package ollamatokenizer provides a simple interface to ollama's internal tokenization functionality.
package ollamatokenizer

import (
	"bytes"
	"fmt"
	"log/slog"
	"os"
	"slices"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	fsggml "github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llama"
	"github.com/ollama/ollama/model"
	_ "github.com/ollama/ollama/model/models" // register all model architectures
	"github.com/ollama/ollama/model/parsers"
	"github.com/ollama/ollama/model/renderers"
	"github.com/ollama/ollama/server"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/thinking"
	"github.com/ollama/ollama/tokenizer"
	modelname "github.com/ollama/ollama/types/model"
)

const errPfx = "ollamatokenizer: "

// ErrNotImplemented is returned when a request uses options that this library
// does not support.
var ErrNotImplemented = fmt.Errorf("not implemented")

// Tokenizer wraps an Ollama tokenizer for a specific model.
type Tokenizer struct {
	engine tokenizer.Tokenizer // native Ollama engine (pure Go)
	llama  *llama.Model        // llama.cpp tokenizer (CGO)
	model  *server.Model       // loaded via server.GetModel()
}

// New creates a Tokenizer for the given model name (e.g. "llama3.2:3b").
// The model must have been pulled via `ollama pull`.
// https://github.com/ollama/ollama/blob/v0.20.5/llm/server.go#L144-L164
func New(name string) (*Tokenizer, error) {
	// https://github.com/ollama/ollama/blob/v0.20.5/server/images.go#L297-L395
	m, err := server.GetModel(name)
	if err != nil {
		return nil, fmt.Errorf(errPfx+"model %q not found (try `ollama pull %s`): %w", name, name, err)
	}

	t := &Tokenizer{model: m}

	f, err := os.Open(m.ModelPath)
	if err != nil {
		return nil, fmt.Errorf(errPfx+"open model file: %w", err)
	}
	defer f.Close()

	ggmlFile, err := fsggml.Decode(f, -1)
	if err != nil {
		return nil, fmt.Errorf(errPfx+"decode GGUF metadata: %w", err)
	}

	// https://github.com/ollama/ollama/blob/v0.20.5/llm/server.go#L148-L164
	var engine tokenizer.Tokenizer
	if envconfig.NewEngine() || ggmlFile.KV().OllamaEngineRequired() {
		tp, tpErr := model.NewTextProcessor(m.ModelPath)
		if tpErr != nil {
			// https://github.com/ollama/ollama/blob/v0.20.5/llm/server.go#L155-L157
			slog.Debug("model not yet supported by Ollama engine, switching to compatibility mode", "model", m.ModelPath, "error", tpErr)
		} else {
			engine = tp
		}
	}
	if engine == nil {
		// https://github.com/ollama/ollama/blob/v0.20.5/llm/server.go#L160
		llamaModel, err := llama.LoadModelFromFile(m.ModelPath, llama.ModelParams{VocabOnly: true})
		if err != nil {
			return nil, fmt.Errorf(errPfx+"llama.cpp tokenizer for %q: %w", name, err)
		}
		t.llama = llamaModel
	} else {
		t.engine = engine
	}
	return t, nil
}

// hasThinking reports whether the model supports thinking.
// https://github.com/ollama/ollama/blob/v0.20.5/server/routes.go#L396
func (t *Tokenizer) hasThinking() bool {
	return slices.Contains(t.model.Capabilities(), modelname.CapabilityThinking)
}

// Tokenize encodes text into token IDs without applying any chat template.
// addSpecial: prepend BOS/append EOS if the model's vocab requires it (AddBOS/AddEOS GGUF metadata).
// parseSpecial: parse special token strings in text (e.g. <|im_start|>) into token IDs.
// https://github.com/ollama/ollama/blob/v0.20.5/runner/ollamarunner/runner.go#L246
// https://github.com/ollama/ollama/blob/v0.20.5/runner/llamarunner/runner.go#L211
func (t *Tokenizer) Tokenize(text string, addSpecial, parseSpecial bool) ([]int32, error) {
	if t.engine != nil {
		tokens, err := t.engine.Encode(text, addSpecial)
		if err != nil {
			return nil, fmt.Errorf(errPfx+"tokenize: %w", err)
		}
		return tokens, nil
	}

	tokens, err := t.llama.Tokenize(text, addSpecial, parseSpecial)
	if err != nil {
		return nil, fmt.Errorf(errPfx+"tokenize: %w", err)
	}
	result := make([]int32, len(tokens))
	for i, tok := range tokens {
		result[i] = int32(tok)
	}
	return result, nil
}

// resolveThink resolves the think parameter, defaulting to true for thinking-capable models.
// https://github.com/ollama/ollama/blob/v0.20.5/server/routes.go#L396-L406
func (t *Tokenizer) resolveThink(think *api.ThinkValue) *api.ThinkValue {
	if think != nil {
		return think
	}
	if t.hasThinking() {
		return &api.ThinkValue{Value: true}
	}
	return nil
}

// renderPrompt renders the prompt via renderer or template.
// https://github.com/ollama/ollama/blob/v0.20.5/server/prompt.go#L116-L136
func (t *Tokenizer) renderPrompt(msgs []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	if t.model.Config.Renderer != "" {
		rendered, err := renderers.RenderWithRenderer(t.model.Config.Renderer, msgs, tools, think)
		if err != nil {
			return "", fmt.Errorf(errPfx+"renderer %q: %w", t.model.Config.Renderer, err)
		}
		return rendered, nil
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

// TokenizeGenerate tokenizes a prompt matching /api/generate.
// Note: the server's chatPrompt performs context-length truncation that we do not replicate;
// for prompts within the context window, the rendered output is identical.
//
// Unsupported options (return ErrNotImplemented): Suffix, Template, Raw, Context, Images.
//
// https://github.com/ollama/ollama/blob/v0.20.5/server/routes.go#L456-L503
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

	think := t.resolveThink(req.Think)
	rendered, err := t.renderPrompt(msgs, nil, think)
	if err != nil {
		return nil, err
	}

	return t.Tokenize(rendered, true, true)
}

// shouldUseHarmony detects harmony-based models.
// https://github.com/ollama/ollama/blob/v0.20.5/server/routes.go#L80-L90
func shouldUseHarmony(m *server.Model) bool {
	if slices.Contains([]string{"gptoss", "gpt-oss"}, m.Config.ModelFamily) {
		if m.Template.Contains("<|start|>") && m.Template.Contains("<|end|>") {
			return true
		}
	}

	return false
}

// processTools initializes the built-in parser and returns processed tools.
// https://github.com/ollama/ollama/blob/v0.20.5/server/routes.go#L2323-L2341
func (t *Tokenizer) processTools(tools []api.Tool, msgs []api.Message, think *api.ThinkValue) []api.Tool {
	// https://github.com/ollama/ollama/blob/v0.20.5/server/routes.go#L2323-L2325
	if shouldUseHarmony(t.model) && t.model.Config.Parser == "" {
		t.model.Config.Parser = "harmony"
	}

	var builtinParser parsers.Parser
	processedTools := tools

	if t.model.Config.Parser != "" {
		builtinParser = parsers.ParserForName(t.model.Config.Parser)
		if builtinParser != nil {
			var lastMessage *api.Message
			if len(msgs) > 0 {
				lastMessage = &msgs[len(msgs)-1]
			}
			processedTools = builtinParser.Init(tools, lastMessage, think)
		}
	}

	return processedTools
}

// filterThinkTags strips thinking content from assistant messages for qwen3 and deepseek-r1 models.
// https://github.com/ollama/ollama/blob/v0.20.5/server/routes.go#L2642-L2668
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

// TokenizeChat tokenizes messages matching /api/chat.
// Note: the server's chatPrompt performs context-length truncation that we do not replicate;
// for prompts within the context window, the rendered output is identical.
//
// https://github.com/ollama/ollama/blob/v0.20.5/server/routes.go#L2276-L2347
func (t *Tokenizer) TokenizeChat(req api.ChatRequest) ([]int32, error) {
	msgs := append(t.model.Messages, req.Messages...)
	if len(req.Messages) > 0 && req.Messages[0].Role != "system" && t.model.System != "" {
		msgs = append([]api.Message{{Role: "system", Content: t.model.System}}, msgs...)
	}
	msgs = filterThinkTags(msgs, t.model)

	// https://github.com/ollama/ollama/blob/v0.20.5/server/routes.go#L2277-L2281
	think := t.resolveThink(req.Think)

	processedTools := t.processTools(req.Tools, msgs, think)

	rendered, err := t.renderPrompt(msgs, processedTools, think)
	if err != nil {
		return nil, err
	}

	return t.Tokenize(rendered, true, true)
}
