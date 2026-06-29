//go:build cgo

// Package ollamatokenizer links ollama's bundled libllama.so and uses llama.cpp's
// real tokenizer with vocab-only model loading. The llama.cpp version must match
// the ollama build (ABI: llama_model_params); pin via go.mod → LLAMA_CPP_VERSION.
// Build inputs (populated by `make fetch-deps`) live under llama-cpp/:
//   - include/llama.h, ggml/include/*.h  (headers)
//   - lib/lib{llama,ggml,ggml-base}.so   (from the ollama install)
package ollamatokenizer

/*
#cgo CFLAGS: -I${SRCDIR}/llama-cpp/include -I${SRCDIR}/llama-cpp/ggml/include
#cgo LDFLAGS: -L${SRCDIR}/llama-cpp/lib -lllama -lggml -lggml-base -lstdc++ -lm
#cgo LDFLAGS: -Wl,-rpath,'${SRCDIR}/llama-cpp/lib'

#include <stdlib.h>
#include "llama.h"

// Wrappers keep llama_model_params layout on the C side; Go uses opaque pointers.

// load_vocab: load only the GGUF vocab (no weights/tensors/GPU).
static void* ot_llama_load_vocab(const char* path) {
	struct llama_model_params p = llama_model_default_params();
	p.vocab_only = true;
	p.n_gpu_layers = 0;
	p.use_mmap = false;
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

// chat_template: the model's GGUF Jinja chat_template (or NULL).
static const char* ot_llama_chat_template(void* model) {
	return llama_model_chat_template((const struct llama_model*) model, NULL);
}

// apply_chat_template: llama.cpp's builtin renderer (no minja). Returns bytes
// written; negative/overflow => retry with larger buf.
static int ot_llama_apply_chat_template(const char* tmpl,
                                        const struct llama_chat_message* msgs, size_t n_msg,
                                        int add_ass, char* buf, int n_max) {
	return (int) llama_chat_apply_template(tmpl, msgs, n_msg, (bool)add_ass, buf, (int32_t)n_max);
}

// add_bos: whether llama.cpp prepends BOS at tokenize(add_special=true); forced
// on at load for some models (e.g. Gemma4) regardless of the GGUF flag.
static int ot_llama_add_bos(const void* vocab) {
	return (int) llama_vocab_get_add_bos((const struct llama_vocab*) vocab);
}
*/
import "C"

import (
	"fmt"
	"unsafe"
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

	m := unsafe.Pointer(C.ot_llama_load_vocab(cpath))
	if m == nil {
		return nil, fmt.Errorf("llama_load_model_from_file(vocab_only) failed for %s", modelPath)
	}
	v := unsafe.Pointer(C.ot_llama_vocab(m))
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

// ChatTemplate returns the model's GGUF Jinja chat template, or "".
func (c *cgoVocab) ChatTemplate() string {
	tmpl := C.ot_llama_chat_template(c.model)
	if tmpl == nil {
		return ""
	}
	return C.GoString(tmpl)
}

// ChatMessage is a {role, content} pair for RenderChat.
type ChatMessage struct {
	Role    string
	Content string
}

// RenderChat applies the model's builtin chat template; addGenerationPrompt
// appends the assistant-turn marker.
func (c *cgoVocab) RenderChat(msgs []ChatMessage, addGenerationPrompt bool) (string, error) {
	tmpl := C.ot_llama_chat_template(c.model)
	if tmpl == nil {
		return "", fmt.Errorf("model has no chat template")
	}
	if len(msgs) == 0 {
		return "", nil
	}
	cMsgs := make([]C.struct_llama_chat_message, len(msgs))
	for i, m := range msgs {
		cMsgs[i] = C.struct_llama_chat_message{
			role:    C.CString(m.Role),
			content: C.CString(m.Content),
		}
	}
	defer func() {
		for i := range cMsgs {
			C.free(unsafe.Pointer(cMsgs[i].role))
			C.free(unsafe.Pointer(cMsgs[i].content))
		}
	}()

	addAss := C.int(0)
	if addGenerationPrompt {
		addAss = 1
	}
	nInt := 1 << 16
	buf := make([]byte, nInt)
	for {
		got := int(C.ot_llama_apply_chat_template(
			tmpl,
			&cMsgs[0],
			C.size_t(len(cMsgs)),
			addAss,
			(*C.char)(unsafe.Pointer(&buf[0])),
			C.int(nInt),
		))
		if got >= 0 && got <= nInt {
			return string(buf[:got]), nil
		}
		nInt = got + 1 // overflow: retry with required size
		buf = make([]byte, nInt)
	}
}

// Encode tokenizes text. addSpecial applies BOS/EOS per the vocab.
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
