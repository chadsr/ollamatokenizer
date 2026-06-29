# CGO build against ollama's bundled libllama.so. llama-cpp/ (headers + libs)
# is gitignored; `make fetch-deps` populates it, pinned to go.mod's ollama version.
#
#   make fetch-deps   # fetch llama.cpp headers + copy ollama libs into llama-cpp/
#   make build        # build the CLI (implies fetch-deps)
#   make tidy         # go mod tidy
#   make clean        # remove bin/
#   make clean-deps   # also remove llama-cpp/

GO ?= go
GOFLAGS ?=
BUILD_DIR := bin

# ollama version from go.mod → llama.cpp version recorded at that tag.
OLLAMA_VERSION := $(shell awk '/github\.com\/ollama\/ollama/ {print $$2}' go.mod)
LLAMA_CPP_VERSION_FILE := llama-cpp/.LLAMA_CPP_VERSION
LLAMA_CPP_VERSION := $(shell cat $(LLAMA_CPP_VERSION_FILE) 2>/dev/null)
ifndef LLAMA_CPP_VERSION
LLAMA_CPP_VERSION := $(shell curl -fsSL https://raw.githubusercontent.com/ollama/ollama/$(OLLAMA_VERSION)/LLAMA_CPP_VERSION 2>/dev/null)
endif

LLAMA_DIR     := llama-cpp
LLAMA_INCLUDE := $(LLAMA_DIR)/include
LLAMA_GGML    := $(LLAMA_DIR)/ggml/include
LLAMA_LIB     := $(LLAMA_DIR)/lib

# Source of libllama.so + libggml*.so: host ollama install by default (matches
# the running server). In Docker: make fetch-deps OLLAMA_LIB_DIR=/ollama-libs
OLLAMA_LIB_DIR ?= /usr/lib/ollama

LLAMA_SRC := https://raw.githubusercontent.com/ggml-org/llama.cpp/$(LLAMA_CPP_VERSION)

LLAMA_HEADERS := \
	$(LLAMA_INCLUDE)/llama.h \
	$(LLAMA_INCLUDE)/llama-cpp.h \
	$(LLAMA_GGML)/ggml.h \
	$(LLAMA_GGML)/ggml-alloc.h \
	$(LLAMA_GGML)/ggml-backend.h \
	$(LLAMA_GGML)/ggml-cpp.h \
	$(LLAMA_GGML)/ggml-cpu.h \
	$(LLAMA_GGML)/ggml-opt.h \
	$(LLAMA_GGML)/gguf.h

.PHONY: all build fetch-deps fetch-headers fetch-libs tidy clean clean-deps

all: build

# Per-file header fetch from llama.cpp at LLAMA_CPP_VERSION.
define fetch_header
$(1): | $(2)
	@curl -fsSL "$(LLAMA_SRC)/$(3)" -o "$$@"
endef

$(LLAMA_INCLUDE) $(LLAMA_GGML) $(LLAMA_LIB):
	@mkdir -p "$@"

$(eval $(call fetch_header,$(LLAMA_INCLUDE)/llama.h,$(LLAMA_INCLUDE),include/llama.h))
$(eval $(call fetch_header,$(LLAMA_INCLUDE)/llama-cpp.h,$(LLAMA_INCLUDE),include/llama-cpp.h))
$(eval $(call fetch_header,$(LLAMA_GGML)/ggml.h,$(LLAMA_GGML),ggml/include/ggml.h))
$(eval $(call fetch_header,$(LLAMA_GGML)/ggml-alloc.h,$(LLAMA_GGML),ggml/include/ggml-alloc.h))
$(eval $(call fetch_header,$(LLAMA_GGML)/ggml-backend.h,$(LLAMA_GGML),ggml/include/ggml-backend.h))
$(eval $(call fetch_header,$(LLAMA_GGML)/ggml-cpp.h,$(LLAMA_GGML),ggml/include/ggml-cpp.h))
$(eval $(call fetch_header,$(LLAMA_GGML)/ggml-cpu.h,$(LLAMA_GGML),ggml/include/ggml-cpu.h))
$(eval $(call fetch_header,$(LLAMA_GGML)/ggml-opt.h,$(LLAMA_GGML),ggml/include/ggml-opt.h))
$(eval $(call fetch_header,$(LLAMA_GGML)/gguf.h,$(LLAMA_GGML),ggml/include/gguf.h))

fetch-headers: $(LLAMA_HEADERS)
	@mkdir -p $(LLAMA_DIR)
	@printf '%s' $(LLAMA_CPP_VERSION) > $(LLAMA_CPP_VERSION_FILE)
	@echo "Headers: llama.cpp $(LLAMA_CPP_VERSION) (via ollama $(OLLAMA_VERSION))"

fetch-libs: | $(LLAMA_LIB)
	@test -f $(OLLAMA_LIB_DIR)/libllama.so || { \
		echo "ollama libs not found in $(OLLAMA_LIB_DIR)"; \
		echo "install ollama, or: make fetch-deps OLLAMA_LIB_DIR=<dir>"; exit 1; }
	cp -a $(OLLAMA_LIB_DIR)/libllama.so*     $(LLAMA_LIB)/
	cp -a $(OLLAMA_LIB_DIR)/libggml.so*      $(LLAMA_LIB)/
	cp -a $(OLLAMA_LIB_DIR)/libggml-base.so* $(LLAMA_LIB)/
	@echo "Libs: copied from $(OLLAMA_LIB_DIR)"

fetch-deps: fetch-headers fetch-libs

tidy:
	$(GO) mod tidy $(GOFLAGS)

build: fetch-deps
	@mkdir -p $(BUILD_DIR)
	CGO_ENABLED=1 $(GO) build -trimpath $(GOFLAGS) -o $(BUILD_DIR)/ollamatokenizer ./cmd/ollamatokenizer

clean:
	rm -rf $(BUILD_DIR)

clean-deps:
	rm -rf $(LLAMA_DIR)
