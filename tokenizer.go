// Package ollamatokenizer provides a simple interface to ollama's internal tokenization functionality.
package ollamatokenizer

import (
	"bytes"
	"fmt"
	"os"
	"slices"

	"github.com/ollama/ollama/api"
	fsggml "github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llama"
	"github.com/ollama/ollama/model"
	_ "github.com/ollama/ollama/model/models" // register all model architectures
	"github.com/ollama/ollama/model/parsers"
	"github.com/ollama/ollama/model/renderers"
	"github.com/ollama/ollama/server"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/tokenizer"
	modelname "github.com/ollama/ollama/types/model"
)

const errPfx = "ollamatokenizer: "

// Tokenizer wraps an Ollama tokenizer for a specific model.
type Tokenizer struct {
	engine tokenizer.Tokenizer // native Ollama engine (pure Go)
	llama  *llama.Model        // llama.cpp tokenizer (CGO)
	model  *server.Model       // loaded via server.GetModel()
}

// New creates a Tokenizer for the given model name (e.g. "llama3.2:3b").
// The model must have been pulled via `ollama pull`.
//
// Uses the same tokenizer selection as the Ollama server:
// OllamaEngineRequired() → native Go; otherwise → llama.cpp.
func New(name string) (*Tokenizer, error) {
	// https://github.com/ollama/ollama/blob/v0.20.5/server/images.go#L297-L395
	m, err := server.GetModel(name)
	if err != nil {
		return nil, fmt.Errorf(errPfx+"model %q not found (try `ollama pull %s`): %w", name, name, err)
	}

	t := &Tokenizer{model: m}

	// Mirrors: https://github.com/ollama/ollama/blob/v0.20.5/llm/server.go#L144-L164
	// envconfig.NewEngine() defaults to false; OllamaEngineRequired() is the deciding factor.
	f, err := os.Open(m.ModelPath)
	if err != nil {
		return nil, fmt.Errorf(errPfx+"open model file: %w", err)
	}
	defer f.Close()

	ggmlFile, err := fsggml.Decode(f, -1)
	if err != nil {
		return nil, fmt.Errorf(errPfx+"decode GGUF metadata: %w", err)
	}

	if ggmlFile.KV().OllamaEngineRequired() {
		// https://github.com/ollama/ollama/blob/v0.20.5/llm/server.go#L150
		tp, err := model.NewTextProcessor(m.ModelPath)
		if err != nil {
			return nil, fmt.Errorf(errPfx+"native tokenizer for %q: %w", name, err)
		}
		t.engine = tp
		return t, nil
	}

	// https://github.com/ollama/ollama/blob/v0.20.5/llm/server.go#L160
	llamaModel, err := llama.LoadModelFromFile(m.ModelPath, llama.ModelParams{VocabOnly: true})
	if err != nil {
		return nil, fmt.Errorf(errPfx+"llama.cpp tokenizer for %q: %w", name, err)
	}
	t.llama = llamaModel
	return t, nil
}

// hasThinking reports whether the model supports thinking.
// Mirrors the server's capability check:
// https://github.com/ollama/ollama/blob/v0.20.5/server/routes.go#L396
// Capabilities() is defined at:
// https://github.com/ollama/ollama/blob/v0.20.5/server/images.go#L80-L155
// Thinking capability is appended at:
// https://github.com/ollama/ollama/blob/v0.20.5/server/images.go#L149-L155
func (t *Tokenizer) hasThinking() bool {
	return slices.Contains(t.model.Capabilities(), modelname.CapabilityThinking)
}

// Tokenize encodes text into token IDs without applying any chat template.
// Matches the runner's tokenization exactly, including BOS/EOS special tokens
// when the model's vocabulary requires them (addSpecial=true).
//
// The runner tokenizes the first text part with addSpecial=true:
//   - llama.cpp path:  https://github.com/ollama/ollama/blob/v0.20.5/runner/llamarunner/runner.go#L211
//   - Ollama engine:   https://github.com/ollama/ollama/blob/v0.20.5/runner/ollamarunner/runner.go#L246
//
// When addSpecial=true, BOS is prepended if the model's vocabulary has AddBOS=true:
//   - llama.cpp: llama_tokenize C function checks add_bos_token GGUF metadata
//   - Ollama engine: vocab.addSpecials() checks v.AddBOS
//     https://github.com/ollama/ollama/blob/v0.20.5/tokenizer/vocabulary.go#L46-L66
//
// EOS is appended if AddEOS=true (rare; most models only set AddBOS).
func (t *Tokenizer) Tokenize(text string) ([]int32, error) {
	if t.engine != nil {
		// https://github.com/ollama/ollama/blob/v0.20.5/runner/ollamarunner/runner.go#L246
		tokens, err := t.engine.Encode(text, true)
		if err != nil {
			return nil, fmt.Errorf(errPfx+"tokenize: %w", err)
		}
		return tokens, nil
	}

	// https://github.com/ollama/ollama/blob/v0.20.5/runner/llamarunner/runner.go#L211
	tokens, err := t.llama.Tokenize(text, true, true)
	if err != nil {
		return nil, fmt.Errorf(errPfx+"tokenize: %w", err)
	}
	result := make([]int32, len(tokens))
	for i, tok := range tokens {
		result[i] = int32(tok)
	}
	return result, nil
}

// resolveThink resolves the think parameter, applying auto-detection for thinking models.
// Returns the resolved *api.ThinkValue (may be nil if thinking is not active).
//
// Mirrors the server's thinking auto-detection:
// https://github.com/ollama/ollama/blob/v0.20.5/server/routes.go#L396-L406
func (t *Tokenizer) resolveThink(think *api.ThinkValue) *api.ThinkValue {
	if think != nil {
		return think
	}
	// https://github.com/ollama/ollama/blob/v0.20.5/server/routes.go#L398-L399
	if t.hasThinking() {
		return &api.ThinkValue{Value: true}
	}
	return nil
}

// renderPrompt renders the prompt via renderer or template.
// Mirrors the server's renderPrompt exactly:
// https://github.com/ollama/ollama/blob/v0.20.5/server/prompt.go#L116-L136
func (t *Tokenizer) renderPrompt(msgs []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	if t.model.Config.Renderer != "" {
		// https://github.com/ollama/ollama/blob/v0.20.5/server/prompt.go#L117-L122
		rendered, err := renderers.RenderWithRenderer(t.model.Config.Renderer, msgs, tools, think)
		if err != nil {
			return "", fmt.Errorf(errPfx+"renderer %q: %w", t.model.Config.Renderer, err)
		}
		return rendered, nil
	}

	// https://github.com/ollama/ollama/blob/v0.20.5/server/prompt.go#L125-L133
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

// TokenizeGenerate tokenizes text matching /api/generate.
// Uses Prompt, System, and Think from api.GenerateRequest.
//
// Message assembly mirrors the generate handler:
// https://github.com/ollama/ollama/blob/v0.20.5/server/routes.go#L456-L471
//
// The server then calls chatPrompt which calls renderPrompt:
// https://github.com/ollama/ollama/blob/v0.20.5/server/routes.go#L497-L503
// https://github.com/ollama/ollama/blob/v0.20.5/server/prompt.go#L23-L113
//
// Note: the server's chatPrompt performs context-length truncation that we do not replicate.
// For prompts within the context window, the rendered output is identical.
// The server also only includes m.Messages when req.Context == nil (deprecated field):
// https://github.com/ollama/ollama/blob/v0.20.5/server/routes.go#L463-L465
func (t *Tokenizer) TokenizeGenerate(req api.GenerateRequest) ([]int32, error) {
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

	return t.Tokenize(rendered)
}

// shouldUseHarmony mirrors the server's harmony detection heuristic exactly:
// https://github.com/ollama/ollama/blob/v0.20.5/server/routes.go#L80-L90
func shouldUseHarmony(m *server.Model) bool {
	if slices.Contains([]string{"gptoss", "gpt-oss"}, m.Config.ModelFamily) {
		// heuristic to check whether the template expects to be parsed via harmony:
		// search for harmony tags that are nearly always used
		if m.Template.Contains("<|start|>") && m.Template.Contains("<|end|>") {
			return true
		}
	}

	return false
}

// processTools mirrors the server's tool processing logic exactly:
// https://github.com/ollama/ollama/blob/v0.20.5/server/routes.go#L2323-L2341
func (t *Tokenizer) processTools(tools []api.Tool, msgs []api.Message, think *api.ThinkValue) []api.Tool {
	// https://github.com/ollama/ollama/blob/v0.20.5/server/routes.go#L2323-L2325
	if shouldUseHarmony(t.model) && t.model.Config.Parser == "" {
		t.model.Config.Parser = "harmony"
	}

	// https://github.com/ollama/ollama/blob/v0.20.5/server/routes.go#L2327-L2341
	var builtinParser parsers.Parser
	processedTools := tools

	if t.model.Config.Parser != "" {
		builtinParser = parsers.ParserForName(t.model.Config.Parser)
		if builtinParser != nil {
			// Determine last message for chat prefill
			var lastMessage *api.Message
			if len(msgs) > 0 {
				lastMessage = &msgs[len(msgs)-1]
			}
			// Initialize parser and get processed tools
			processedTools = builtinParser.Init(tools, lastMessage, think)
		}
	}

	return processedTools
}

// TokenizeChat tokenizes messages matching /api/chat.
// Uses Messages, Tools, and Think from api.ChatRequest.
//
// Message assembly mirrors the chat handler:
// https://github.com/ollama/ollama/blob/v0.20.5/server/routes.go#L2317-L2321
//
// Tool processing mirrors the chat handler:
// https://github.com/ollama/ollama/blob/v0.20.5/server/routes.go#L2323-L2341
//
// The server then calls chatPrompt which calls renderPrompt:
// https://github.com/ollama/ollama/blob/v0.20.5/server/routes.go#L2347
// https://github.com/ollama/ollama/blob/v0.20.5/server/prompt.go#L23-L113
//
// Note: the server calls filterThinkTags for qwen3/deepseek-r1 models to strip thinking
// content from assistant messages before the final user message:
// https://github.com/ollama/ollama/blob/v0.20.5/server/routes.go#L2321
// https://github.com/ollama/ollama/blob/v0.20.5/server/routes.go#L2642-L2668
// Our library does not replicate this filtering, which may cause token count differences
// for those specific models when assistant messages contain thinking content.
func (t *Tokenizer) TokenizeChat(req api.ChatRequest) ([]int32, error) {
	msgs := append(t.model.Messages, req.Messages...)
	if len(req.Messages) > 0 && req.Messages[0].Role != "system" && t.model.System != "" {
		msgs = append([]api.Message{{Role: "system", Content: t.model.System}}, msgs...)
	}

	processedTools := t.processTools(req.Tools, msgs, req.Think)

	think := t.resolveThink(req.Think)
	rendered, err := t.renderPrompt(msgs, processedTools, think)
	if err != nil {
		return nil, err
	}

	return t.Tokenize(rendered)
}
