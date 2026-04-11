// Package ollamatokenizer provides 1:1 tokenization with a running Ollama instance.
//
// No model weights are loaded and no inference is performed — only tokenizer
// metadata is read from GGUF files. Chat templates and model config are loaded
// via server.GetModel(). Set OLLAMA_MODELS to your model directory.
//
// Building requires CGO and a C/C++ toolchain (llama.cpp tokenizer).
//
// # Tokenize (raw, no template)
//
//	tok, _ := ollamatokenizer.New("llama3.2:3b")
//	tokens, _ := tok.Tokenize("hello world")
//
// # TokenizeGenerate (applies chat template, matching /api/generate)
//
//	tokens, _ := tok.TokenizeGenerate(api.GenerateRequest{Prompt: "hello world"})
//
// # TokenizeChat (applies chat template, matching /api/chat)
//
//	tokens, _ := tok.TokenizeChat(api.ChatRequest{
//	    Messages: []api.Message{{Role: "user", Content: "hello world"}},
//	})
package ollamatokenizer

import (
	"bytes"
	"fmt"
	"os"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
	fsggml "github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llama"
	"github.com/ollama/ollama/model"
	_ "github.com/ollama/ollama/model/models" // register all model architectures
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
// https://github.com/ollama/ollama/blob/v0.20.5/server/images.go#L149-L155
func (t *Tokenizer) hasThinking() bool {
	return slices.Contains(t.model.Capabilities(), modelname.CapabilityThinking)
}

// Tokenize encodes text into token IDs without applying any chat template.
func (t *Tokenizer) Tokenize(text string) ([]int32, error) {
	if t.engine != nil {
		tokens, err := t.engine.Encode(text, false)
		if err != nil {
			return nil, fmt.Errorf(errPfx+"tokenize: %w", err)
		}
		return tokens, nil
	}

	tokens, err := t.llama.Tokenize(text, false, true)
	if err != nil {
		return nil, fmt.Errorf(errPfx+"tokenize: %w", err)
	}
	result := make([]int32, len(tokens))
	for i, tok := range tokens {
		result[i] = int32(tok)
	}
	return result, nil
}

// Detokenize decodes token IDs back into a string.
func (t *Tokenizer) Detokenize(tokens []int32) (string, error) {
	if t.engine != nil {
		text, err := t.engine.Decode(tokens)
		if err != nil {
			return "", fmt.Errorf(errPfx+"detokenize: %w", err)
		}
		return text, nil
	}

	var sb strings.Builder
	for _, tok := range tokens {
		sb.WriteString(t.llama.TokenToPiece(int(tok)))
	}
	return sb.String(), nil
}

// TokenizeToStrings returns the string representation of each token.
func (t *Tokenizer) TokenizeToStrings(text string) ([]string, error) {
	tokens, err := t.Tokenize(text)
	if err != nil {
		return nil, err
	}

	if t.engine != nil {
		vocab := t.engine.Vocabulary()
		result := make([]string, len(tokens))
		for i, id := range tokens {
			if int(id) < len(vocab.Values) {
				result[i] = vocab.Values[id]
			} else {
				result[i] = fmt.Sprintf("<unk:%d>", id)
			}
		}
		return result, nil
	}

	result := make([]string, len(tokens))
	for i, id := range tokens {
		result[i] = t.llama.TokenToPiece(int(id))
	}
	return result, nil
}

// resolveThink mirrors thinking auto-detection.
// https://github.com/ollama/ollama/blob/v0.20.5/server/routes.go#L396-L400
// https://github.com/ollama/ollama/blob/v0.20.5/server/routes.go#L474-L479
func (t *Tokenizer) resolveThink(think *api.ThinkValue) (thinkVal bool, thinkLevel string, isThinkSet bool) {
	isThinkSet = think != nil
	if think != nil {
		thinkVal = think.Bool()
		thinkLevel = think.String()
	} else if t.hasThinking() {
		// Server sets req.Think = &ThinkValue{Value: true}, making both Think=true and IsThinkSet=true.
		thinkVal = true
		isThinkSet = true
	}
	return
}

// renderPrompt renders the prompt via renderer or template.
// https://github.com/ollama/ollama/blob/v0.20.5/server/prompt.go#L116-L136
func (t *Tokenizer) renderPrompt(msgs []api.Message, thinkVal bool, thinkLevel string, isThinkSet bool) (string, error) {
	if t.model.Config.Renderer != "" {
		// https://github.com/ollama/ollama/blob/v0.20.5/server/prompt.go#L117-L123
		think := &api.ThinkValue{}
		if thinkVal {
			think.Value = true
		}
		if !isThinkSet {
			think = nil
		}
		rendered, err := renderers.RenderWithRenderer(t.model.Config.Renderer, msgs, nil, think)
		if err != nil {
			return "", fmt.Errorf(errPfx+"renderer %q: %w", t.model.Config.Renderer, err)
		}
		return rendered, nil
	}

	// https://github.com/ollama/ollama/blob/v0.20.5/server/prompt.go#L125-L133
	var b bytes.Buffer
	if err := t.model.Template.Execute(&b, template.Values{
		Messages:   msgs,
		Think:      thinkVal,
		ThinkLevel: thinkLevel,
		IsThinkSet: isThinkSet,
	}); err != nil {
		return "", fmt.Errorf(errPfx+"template: %w", err)
	}
	return b.String(), nil
}

// TokenizeGenerate tokenizes text matching /api/generate.
// Uses Prompt, System, and Think from api.GenerateRequest.
func (t *Tokenizer) TokenizeGenerate(req api.GenerateRequest) ([]int32, error) {
	// https://github.com/ollama/ollama/blob/v0.20.5/server/routes.go#L456-L471
	var msgs []api.Message
	if req.System != "" {
		msgs = append(msgs, api.Message{Role: "system", Content: req.System})
	} else if t.model.System != "" {
		msgs = append(msgs, api.Message{Role: "system", Content: t.model.System})
	}

	msgs = append(msgs, t.model.Messages...)
	msgs = append(msgs, api.Message{Role: "user", Content: req.Prompt})

	thinkVal, thinkLevel, isThinkSet := t.resolveThink(req.Think)
	rendered, err := t.renderPrompt(msgs, thinkVal, thinkLevel, isThinkSet)
	if err != nil {
		return nil, err
	}

	return t.Tokenize(rendered)
}

// TokenizeChat tokenizes messages matching /api/chat.
// Uses Messages and Think from api.ChatRequest.
func (t *Tokenizer) TokenizeChat(req api.ChatRequest) ([]int32, error) {
	// https://github.com/ollama/ollama/blob/v0.20.5/server/routes.go#L2317-L2321
	msgs := append(t.model.Messages, req.Messages...)
	if len(req.Messages) > 0 && req.Messages[0].Role != "system" && t.model.System != "" {
		msgs = append([]api.Message{{Role: "system", Content: t.model.System}}, msgs...)
	}

	thinkVal, thinkLevel, isThinkSet := t.resolveThink(req.Think)
	rendered, err := t.renderPrompt(msgs, thinkVal, thinkLevel, isThinkSet)
	if err != nil {
		return nil, err
	}

	return t.Tokenize(rendered)
}
