//go:build cgo

// Package ollamatokenizer: cgo bindings for llama.cpp tokenization.
package ollamatokenizer

/*
#cgo CFLAGS: -I${SRCDIR}/llama-cpp/include -I${SRCDIR}/llama-cpp/ggml/include -I${SRCDIR}/llama-cpp
#cgo LDFLAGS: -L${SRCDIR}/llama-cpp/lib -lotchat -lllama -lllama-common -lggml -lggml-base -lstdc++ -lm
#cgo LDFLAGS: -Wl,-rpath,'${SRCDIR}/llama-cpp/lib'

#include <stdlib.h>
#include "llama.h"

// ot_chat_apply: jinja/shim.cpp → libotchat.a → libllama-common.so.
extern int ot_chat_apply(const void* model, const char* messages_json, const char* tools_json,
    int think_set, int enable_thinking, const char* reasoning_effort,
    char* buf, int buf_len);

// Wrappers keep llama_model_params layout on the C side; Go uses opaque pointers.

// load_vocab: vocab-only load, no weights/tensors/GPU.
static void* ot_llama_load_vocab(const char* path) {
	struct llama_model_params p = llama_model_default_params();
	p.vocab_only = true;
	p.n_gpu_layers = 0;
	return (void*) llama_model_load_from_file(path, p);
}
static const void* ot_llama_vocab(void* model) {
	return (const void*) llama_model_get_vocab((const struct llama_model*) model);
}
static void ot_llama_free(void* model) {
	llama_model_free((struct llama_model*) model);
}

// tokenize: on success returns token count; negative => |-n| is required buffer.
static int ot_llama_tokenize(const void* vocab, const char* text, int text_len,
                             int* out, int n_max, int add_special, int parse_special) {
	return (int) llama_tokenize((const struct llama_vocab*) vocab, text, (int32_t)text_len,
	                            (llama_token*) out, (int32_t)n_max,
	                            (bool)add_special, (bool)parse_special);
}

// silence: install a no-op log callback to keep vocab-load chatter off stderr.
static void ot_llama_silence_cb(enum ggml_log_level level, const char* text, void* user_data) {
	(void) level; (void) text; (void) user_data;
}
static void ot_llama_silence(void) {
	llama_log_set(ot_llama_silence_cb, NULL);
}

// add_bos: whether tokenize(add_special=true) prepends BOS. Matches ollama's
// tokenizerAddsBOS() — llama.cpp forces add_bos for lfm2/gemma4 at load time.
// https://github.com/ollama/ollama/blob/v0.40.0/llm/llama_server.go#L260-L280
static int ot_llama_add_bos(const void* vocab) {
	return (int) llama_vocab_get_add_bos((const struct llama_vocab*) vocab);
}
*/
import "C"

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"unsafe"

	"github.com/ollama/ollama/api"
)

func init() {
	C.ot_llama_silence()
}

// cgoVocab wraps a vocab-only llama.cpp model handle.
type cgoVocab struct {
	model unsafe.Pointer // llama_model*
	vocab unsafe.Pointer // llama_vocab* (borrowed from model)
}

func newCGOVocab(modelPath string) (*cgoVocab, error) {
	cpath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cpath))

	m := C.ot_llama_load_vocab(cpath)
	if m == nil {
		return nil, fmt.Errorf("llama_load_model_from_file(vocab_only) failed for %s", modelPath)
	}
	v := C.ot_llama_vocab(m)
	if v == nil {
		C.ot_llama_free(m)
		return nil, fmt.Errorf("model %s has no vocabulary", modelPath)
	}
	return &cgoVocab{model: m, vocab: v}, nil
}

func (c *cgoVocab) Close() {
	if c != nil && c.model != nil {
		C.ot_llama_free(c.model)
		c.model = nil
		c.vocab = nil
	}
}

// Encode tokenizes text.
func (c *cgoVocab) Encode(text string, addSpecial, parseSpecial bool) ([]int32, error) {
	var cText *C.char
	if len(text) > 0 {
		cText = (*C.char)(unsafe.Pointer(unsafe.StringData(text)))
	} else {
		cText = (*C.char)(unsafe.Pointer(&empty[0]))
	}
	buf := make([]int32, len(text)+8)
	for {
		n := C.ot_llama_tokenize(
			c.vocab,
			cText,
			C.int(len(text)),
			(*C.int)(unsafe.Pointer(&buf[0])),
			C.int(len(buf)),
			cbool(addSpecial),
			cbool(parseSpecial),
		)
		switch {
		case n >= 0:
			return buf[:n:n], nil
		case n == -1:
			return nil, fmt.Errorf("llama_tokenize: invalid input")
		default:
			need := int(-n) + 1
			if need <= len(buf) {
				return nil, fmt.Errorf("llama_tokenize: requested %d but have %d", need, len(buf))
			}
			buf = make([]int32, need)
		}
	}
}

// AddBOS reports whether llama.cpp prepends BOS at tokenize(add_special=true).
func (c *cgoVocab) AddBOS() bool { return C.ot_llama_add_bos(c.vocab) != 0 }

func cbool(b bool) C.int {
	if b {
		return 1
	}
	return 0
}

var empty [1]byte // sentinel pointer for empty input

// serverMessage is the message JSON ollama posts to llama-server.
// https://github.com/ollama/ollama/blob/v0.40.0/llm/llama_server.go#L2312-L2360
type serverMessage struct {
	Role       string           `json:"role"`
	Content    string           `json:"content"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
	Name       string           `json:"name,omitempty"`
	ToolCalls  []serverToolCall `json:"tool_calls,omitempty"`
}

// serverToolCall mirrors llamaServerChatToolCall.
// https://github.com/ollama/ollama/blob/v0.40.0/llm/llama_server.go#L2197-L2205
type serverToolCall struct {
	ID       string `json:"id,omitempty"`
	Index    int    `json:"index"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// serverMessages marshals messages like llamaServerChatMessage.
// https://github.com/ollama/ollama/blob/v0.40.0/llm/llama_server.go#L2304-L2330
func serverMessages(msgs []api.Message) ([]serverMessage, error) {
	out := make([]serverMessage, len(msgs))
	for i, m := range msgs {
		sm := serverMessage{
			Role:       m.Role,
			Content:    m.Content,
			ToolCallID: m.ToolCallID,
			Name:       m.ToolName,
		}
		for _, tc := range m.ToolCalls {
			args, err := json.Marshal(tc.Function.Arguments)
			if err != nil {
				return nil, fmt.Errorf("marshal tool call arguments for %q: %w", tc.Function.Name, err)
			}
			var stc serverToolCall
			stc.ID = tc.ID
			stc.Index = tc.Function.Index
			stc.Type = "function"
			stc.Function.Name = tc.Function.Name
			stc.Function.Arguments = string(args)
			sm.ToolCalls = append(sm.ToolCalls, stc)
		}
		out[i] = sm
	}
	return out, nil
}

// RenderChatJinja applies the GGUF chat template via the shim.
// https://github.com/ollama/ollama/blob/v0.40.0/llm/llama_server.go#L1945-L1986
func (c *cgoVocab) RenderChatJinja(msgs []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	messages, err := serverMessages(msgs)
	if err != nil {
		return "", err
	}
	messagesJSON, err := json.Marshal(messages)
	if err != nil {
		return "", fmt.Errorf("marshal messages: %w", err)
	}
	cMessages := C.CString(string(messagesJSON))
	defer C.free(unsafe.Pointer(cMessages))

	// Tools pass through verbatim, like ollama's body["tools"] = req.Tools.
	var cTools *C.char
	if len(tools) > 0 {
		toolsJSON, err := json.Marshal(tools)
		if err != nil {
			return "", fmt.Errorf("marshal tools: %w", err)
		}
		cTools = C.CString(string(toolsJSON))
		defer C.free(unsafe.Pointer(cTools))
	}

	// llamaServerChatTemplateKwargs: kwargs only when think is set.
	// https://github.com/ollama/ollama/blob/v0.40.0/llm/llama_server.go#L2296-L2310
	thinkSet, enableThinking := 0, 0
	var cEffort *C.char
	if think != nil {
		thinkSet = 1
		if think.Bool() {
			enableThinking = 1
		}
		if think.IsString() {
			cEffort = C.CString(think.String())
			defer C.free(unsafe.Pointer(cEffort))
		}
	}

	n := 1 << 16
	buf := make([]byte, n)
	for {
		got := int(C.ot_chat_apply(
			c.model,
			cMessages,
			cTools,
			C.int(thinkSet),
			C.int(enableThinking),
			cEffort,
			(*C.char)(unsafe.Pointer(&buf[0])),
			C.int(n),
		))
		switch {
		case got >= 0:
			return string(buf[:got]), nil
		case got == -1, got == -2:
			// -2 marks invalid-input errors (HTTP 400); message is in buf.
			msg := strings.TrimSpace(cstring(buf))
			if msg == "" {
				msg = "chat template apply failed"
			}
			if got == -2 {
				return "", &BadRequestError{Err: errors.New(msg)}
			}
			return "", errors.New(msg)
		default:
			need := -got
			if need <= n {
				return "", fmt.Errorf("chat apply: requested %d but have %d", need, n)
			}
			n = need
			buf = make([]byte, n)
		}
	}
}

// cstring reads a NUL-terminated string out of a byte buffer.
func cstring(buf []byte) string {
	if i := bytes.IndexByte(buf, 0); i >= 0 {
		return string(buf[:i])
	}
	return string(buf)
}
