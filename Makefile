# CGO build against ollama's bundled libllama.so. llama-cpp/ (headers + libs)
# is gitignored; `make fetch-deps` populates it, pinned to go.mod's ollama version.
#
#   make fetch-deps    # fetch llama.cpp headers + jinja sources + copy ollama libs
#   make build         # build the CLI (implies fetch-deps + build-jinja)
#   make docker-build  # build the Docker image (ollama version pinned via go.mod)
#   make tidy          # go mod tidy
#   make clean         # remove bin/
#   make clean-deps    # also remove llama-cpp/

GO ?= go
GOFLAGS ?=
BUILD_DIR := bin
DOCKER_IMAGE ?= ollamatokenizer

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

JINJA_DIR := $(LLAMA_DIR)/jinja
JINJA_HEADERS := $(JINJA_DIR)/utils.h $(JINJA_DIR)/string.h $(JINJA_DIR)/lexer.h \
	$(JINJA_DIR)/value.h $(JINJA_DIR)/runtime.h $(JINJA_DIR)/parser.h \
	$(JINJA_DIR)/caps.h $(JINJA_DIR)/unicode.h
JINJA_SOURCES := $(JINJA_DIR)/string.cpp $(JINJA_DIR)/lexer.cpp $(JINJA_DIR)/value.cpp \
	$(JINJA_DIR)/runtime.cpp $(JINJA_DIR)/parser.cpp $(JINJA_DIR)/caps.cpp \
	$(JINJA_DIR)/unicode.cpp $(JINJA_DIR)/wrapper.cpp
JINJA_OBJECTS := $(JINJA_SOURCES:.cpp=.o)

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

.PHONY: all build fetch-deps fetch-headers fetch-jinja fetch-libs build-jinja docker-build tidy clean clean-deps

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

# Jinja (minja) sources from llama.cpp's common/jinja/ - the same Jinja engine
# ollama's llama-server uses for chat template rendering.
fetch-jinja: $(JINJA_HEADERS) $(JINJA_SOURCES)
	@echo "Jinja: fetched from llama.cpp $(LLAMA_CPP_VERSION)"

$(JINJA_DIR):
	@mkdir -p "$@"

# Fetch jinja header/source files from llama.cpp.
$(JINJA_DIR)/%.h $(JINJA_DIR)/%.cpp &: | $(JINJA_DIR)
	@for f in utils.h string.h string.cpp lexer.h lexer.cpp value.h value.cpp \
	          runtime.h runtime.cpp parser.h parser.cpp caps.h caps.cpp; do \
		curl -fsSL "$(LLAMA_SRC)/common/jinja/$$f" -o "$(JINJA_DIR)/$$f" || exit 1; \
	done
	@curl -fsSL "$(LLAMA_SRC)/common/unicode.h" -o "$(JINJA_DIR)/unicode.h" || exit 1
	@curl -fsSL "$(LLAMA_SRC)/common/unicode.cpp" -o "$(JINJA_DIR)/unicode.cpp" || exit 1

# wrapper.cpp is our own — copy from the repo's jinja/ dir.
$(JINJA_DIR)/wrapper.cpp: | $(JINJA_DIR)
	@cp jinja/wrapper.cpp $(JINJA_DIR)/

# Compile jinja sources into a static library. Uses -include cstring to avoid
# the string.h / <string.h> filename collision with the system header.
$(JINJA_DIR)/%.o: $(JINJA_DIR)/%.cpp $(JINJA_HEADERS)
	g++ -std=c++17 -O2 -c -include cstring -I$(LLAMA_DIR) "$<" -o "$@"

build-jinja: $(JINJA_OBJECTS)
	@ar rcs $(LLAMA_LIB)/libotjinja.a $(JINJA_OBJECTS)
	@echo "Jinja: libotjinja.a built"

fetch-libs: | $(LLAMA_LIB)
	@test -f $(OLLAMA_LIB_DIR)/libllama.so || { \
		echo "ollama libs not found in $(OLLAMA_LIB_DIR)"; \
		echo "install ollama, or: make fetch-deps OLLAMA_LIB_DIR=<dir>"; exit 1; }
	cp -a $(OLLAMA_LIB_DIR)/libllama.so*     $(LLAMA_LIB)/
	cp -a $(OLLAMA_LIB_DIR)/libggml.so*      $(LLAMA_LIB)/
	cp -a $(OLLAMA_LIB_DIR)/libggml-base.so* $(LLAMA_LIB)/
	@echo "Libs: copied from $(OLLAMA_LIB_DIR)"

fetch-deps: fetch-headers fetch-jinja fetch-libs

tidy:
	$(GO) mod tidy $(GOFLAGS)

build: fetch-deps build-jinja
	@mkdir -p $(BUILD_DIR)
	CGO_ENABLED=1 $(GO) build -trimpath $(GOFLAGS) -o $(BUILD_DIR)/ollamatokenizer ./cmd/ollamatokenizer

# Single-sourced Docker build: OLLAMA_VERSION comes from go.mod and is passed as
# a build-arg (the Dockerfile has no default - Docker can't read go.mod to resolve
# the FROM tag at parse time). CI does the same derivation.
docker-build:
	docker build --build-arg OLLAMA_VERSION=$(OLLAMA_VERSION) -t $(DOCKER_IMAGE) .

clean:
	rm -rf $(BUILD_DIR)

clean-deps:
	rm -rf $(LLAMA_DIR)
