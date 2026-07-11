//go:build cgo

// Package ollamatokenizer links ollama's bundled libllama.so and uses llama.cpp's
// real tokenizer with vocab-only model loading. The llama.cpp version must match
// the ollama build (ABI: llama_model_params); pin via go.mod → LLAMA_CPP_VERSION.
// Build inputs (populated by `make fetch-deps`) live under llama-cpp/:
//   - include/llama.h, ggml/include/*.h  (headers)
//   - lib/lib{llama,ggml,ggml-base}.so   (from the ollama install)
package ollamatokenizer

/*
#cgo CFLAGS: -I${SRCDIR}/llama-cpp/include -I${SRCDIR}/llama-cpp/ggml/include -I${SRCDIR}/llama-cpp
#cgo LDFLAGS: -L${SRCDIR}/llama-cpp/lib -lllama -lggml -lggml-base -lotjinja -lstdc++ -lm
#cgo LDFLAGS: -Wl,-rpath,'${SRCDIR}/llama-cpp/lib'

#include <stdlib.h>
#include "llama.h"

struct ot_jinja_message { const char *role; const char *content; };
extern int ot_jinja_render(const char *tmpl, const struct ot_jinja_message *msgs, int n_msgs,
    const char *bos_token, const char *eos_token, int add_generation_prompt,
    char *buf, int buf_len);

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

// add_bos: whether llama.cpp prepends BOS at tokenize(add_special=true).
static int ot_llama_add_bos(const void* vocab) {
	return (int) llama_vocab_get_add_bos((const struct llama_vocab*) vocab);
}

// bos_id / bos_piece: the vocab's BOS token, for restoring a {{ bos_token }}
// the builtin renderer drops on native-Jinja templates.
static int ot_llama_bos_id(const void* vocab) {
	return (int) llama_vocab_bos((const struct llama_vocab*) vocab);
}
static int ot_llama_token_piece(const void* vocab, int token, char* buf, int n) {
	return (int) llama_token_to_piece((const struct llama_vocab*) vocab, (llama_token) token, buf, (int32_t)n, 0, true);
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

// ChatTemplate returns the model's GGUF Jinja chat template, or "".
func (c *cgoVocab) ChatTemplate() string {
	tmpl := C.ot_llama_chat_template(c.model)
	if tmpl == nil {
		return ""
	}
	return C.GoString(tmpl)
}

// ChatMessage is a {role, content} pair for RenderChatJinja.
type ChatMessage struct {
	Role    string
	Content string
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

// BOSPiece returns the textual piece for the vocab's BOS token.
func (c *cgoVocab) BOSPiece() string {
	id := C.ot_llama_bos_id(c.vocab)
	if id < 0 {
		return ""
	}
	buf := make([]byte, 64)
	n := int(C.ot_llama_token_piece(c.vocab, id, (*C.char)(unsafe.Pointer(&buf[0])), C.int(len(buf))))
	if n <= 0 {
		return ""
	}
	if n > len(buf) {
		buf = make([]byte, n)
		n = int(C.ot_llama_token_piece(c.vocab, id, (*C.char)(unsafe.Pointer(&buf[0])), C.int(len(buf))))
	}
	return string(buf[:n])
}

func cbool(b bool) C.int {
	if b {
		return 1
	}
	return 0
}

var empty [1]byte // sentinel pointer for empty input

// RenderChatJinja renders messages through the model's Jinja chat template using
// llama.cpp's minja engine - the same engine ollama uses.
func (c *cgoVocab) RenderChatJinja(msgs []ChatMessage, bosToken, eosToken string, addGenerationPrompt bool) (string, error) {
	tmpl := C.ot_llama_chat_template(c.model)
	if tmpl == nil {
		return "", fmt.Errorf("model has no chat template")
	}

	cMsgs := make([]C.struct_ot_jinja_message, len(msgs))
	for i, m := range msgs {
		cMsgs[i] = C.struct_ot_jinja_message{
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

	var bosC, eosC *C.char
	if bosToken != "" {
		bosC = C.CString(bosToken)
		defer C.free(unsafe.Pointer(bosC))
	}
	if eosToken != "" {
		eosC = C.CString(eosToken)
		defer C.free(unsafe.Pointer(eosC))
	}

	n := 1 << 16
	buf := make([]byte, n)
	for {
		got := int(C.ot_jinja_render(
			tmpl,
			&cMsgs[0],
			C.int(len(cMsgs)),
			bosC,
			eosC,
			cbool(addGenerationPrompt),
			(*C.char)(unsafe.Pointer(&buf[0])),
			C.int(n),
		))
		if got >= 0 {
			return string(buf[:got]), nil
		}
		if got == -1 {
			return "", fmt.Errorf("jinja template render failed")
		}
		n = -got
		buf = make([]byte, n)
	}
}
