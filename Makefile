# Requires CGO (llama.cpp tokenizer). The include/ directory contains C++ headers
# stripped by the Go module proxy. Run "make update-headers" after changing go.mod.
#
#   make build            # Build the CLI tool
#   make update-headers   # Re-download C++ headers after changing go.mod

GO ?= go
GOFLAGS ?=
BUILD_DIR := bin

OLLAMA_VERSION := $(shell grep 'github.com/ollama/ollama' go.mod | awk '{print $$2}')
INCLUDE_DIR := include

HEADER_FILES := \
	$(INCLUDE_DIR)/nlohmann/json.hpp \
	$(INCLUDE_DIR)/nlohmann/json_fwd.hpp \
	$(INCLUDE_DIR)/stb/stb_image.h \
	$(INCLUDE_DIR)/miniaudio/miniaudio.h

GITHUB_BASE := https://raw.githubusercontent.com/ollama/ollama/$(OLLAMA_VERSION)

# Satisfies #include "nlohmann/json.hpp" etc. that the module cache lacks.
CGO_CPPFLAGS := -I$(CURDIR)/$(INCLUDE_DIR)

.PHONY: all build update-headers tidy clean

all: build

$(INCLUDE_DIR)/nlohmann/json.hpp: | $(INCLUDE_DIR)/nlohmann
	@curl -fsSL "$(GITHUB_BASE)/llama/llama.cpp/vendor/nlohmann/json.hpp" -o "$@"

$(INCLUDE_DIR)/nlohmann/json_fwd.hpp: | $(INCLUDE_DIR)/nlohmann
	@curl -fsSL "$(GITHUB_BASE)/llama/llama.cpp/vendor/nlohmann/json_fwd.hpp" -o "$@"

$(INCLUDE_DIR)/stb/stb_image.h: | $(INCLUDE_DIR)/stb
	@curl -fsSL "$(GITHUB_BASE)/llama/llama.cpp/vendor/stb/stb_image.h" -o "$@"

$(INCLUDE_DIR)/miniaudio/miniaudio.h: | $(INCLUDE_DIR)/miniaudio
	@curl -fsSL "$(GITHUB_BASE)/llama/llama.cpp/vendor/miniaudio/miniaudio.h" -o "$@"

$(INCLUDE_DIR)/nlohmann $(INCLUDE_DIR)/stb $(INCLUDE_DIR)/miniaudio:
	@mkdir -p "$@"

update-headers: $(HEADER_FILES)
	@echo "Headers updated for ollama $(OLLAMA_VERSION)"

tidy:
	$(GO) mod tidy $(GOFLAGS)

build: $(HEADER_FILES)
	@mkdir -p $(BUILD_DIR)
	CGO_ENABLED=1 CGO_CPPFLAGS="$(CGO_CPPFLAGS)" $(GO) build $(GOFLAGS) -o $(BUILD_DIR)/ollamatokenizer ./cmd/ollamatokenizer

clean:
	rm -rf $(BUILD_DIR)
