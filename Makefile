# ollamatokenizer - Build configuration
#
# This package requires CGO and a C/C++ toolchain because it uses the llama.cpp
# tokenizer (via ollama's llama package) for most model architectures.
#
# The Go module proxy strips vendor/ directories from module zips, which means
# the C++ header-only libraries that llama.cpp needs (nlohmann/json, stb_image,
# miniaudio) are missing. The Makefile's "vendor" target downloads these from
# the ollama GitHub repo at the exact version pinned in go.mod and installs
# them into the Go module cache.
#
# Quick start:
#   make vendor         # Download C++ vendor headers (run once, or after changing go.mod)
#   make build          # Build the CLI tool

GO ?= go
GOFLAGS ?=
BUILD_DIR := bin

# Extract the pinned ollama version from go.mod
OLLAMA_VERSION := $(shell grep 'github.com/ollama/ollama' go.mod | awk '{print $$2}')

# Paths into the Go module cache where CGO expects the vendor headers
MODCACHE := $(shell $(GO) env GOMODCACHE)
OLLAMA_MODDIR := $(MODCACHE)/github.com/ollama/ollama@$(OLLAMA_VERSION)
LLAMA_VENDOR := $(OLLAMA_MODDIR)/llama/llama.cpp/vendor

# The 4 header-only C++ libraries that llama.cpp needs
VENDOR_FILES := \
	$(LLAMA_VENDOR)/nlohmann/json.hpp \
	$(LLAMA_VENDOR)/nlohmann/json_fwd.hpp \
	$(LLAMA_VENDOR)/stb/stb_image.h \
	$(LLAMA_VENDOR)/miniaudio/miniaudio.h

# GitHub raw content base URL for the ollama repo at the pinned version
GITHUB_BASE := https://raw.githubusercontent.com/ollama/ollama/$(OLLAMA_VERSION)

.PHONY: all vendor build tidy clean

all: vendor build

# vendor downloads the C++ header-only libraries that the Go module proxy
# strips out. These are fetched from the ollama GitHub repo at the exact
# version pinned in go.mod, so they are always in sync.
#
# The Go module cache is read-only, so we chmod +w the target directories
# before writing. This is safe — the module cache contents are integrity-
# checked via go.sum, and we're only adding files that should have been there.
vendor: ensure-download
	@echo "Installing vendor headers into module cache..."
	@chmod +w $(OLLAMA_MODDIR)/llama/llama.cpp
	@mkdir -p $(LLAMA_VENDOR)/nlohmann $(LLAMA_VENDOR)/stb $(LLAMA_VENDOR)/miniaudio
	@chmod +w $(LLAMA_VENDOR) $(LLAMA_VENDOR)/nlohmann $(LLAMA_VENDOR)/stb $(LLAMA_VENDOR)/miniaudio
	@curl -fsSL "$(GITHUB_BASE)/llama/llama.cpp/vendor/nlohmann/json.hpp" -o "$(LLAMA_VENDOR)/nlohmann/json.hpp"
	@curl -fsSL "$(GITHUB_BASE)/llama/llama.cpp/vendor/nlohmann/json_fwd.hpp" -o "$(LLAMA_VENDOR)/nlohmann/json_fwd.hpp"
	@curl -fsSL "$(GITHUB_BASE)/llama/llama.cpp/vendor/stb/stb_image.h" -o "$(LLAMA_VENDOR)/stb/stb_image.h"
	@curl -fsSL "$(GITHUB_BASE)/llama/llama.cpp/vendor/miniaudio/miniaudio.h" -o "$(LLAMA_VENDOR)/miniaudio/miniaudio.h"
	@echo "Vendor headers installed for $(OLLAMA_VERSION)"

# ensure-download verifies the ollama module is present in the cache
ensure-download:
ifndef OLLAMA_VERSION
	$(error Could not detect ollama version from go.mod)
endif
	@if [ ! -d "$(OLLAMA_MODDIR)" ]; then \
		echo "Downloading github.com/ollama/ollama@$(OLLAMA_VERSION) into module cache..."; \
		$(GO) mod download github.com/ollama/ollama@$(OLLAMA_VERSION); \
	fi
	@if [ -f "$(LLAMA_VENDOR)/nlohmann/json.hpp" ]; then \
		echo "Vendor headers already present for ollama $(OLLAMA_VERSION)"; \
	fi

# tidy resolves dependencies and generates go.sum
tidy:
	$(GO) mod tidy $(GOFLAGS)

# build compiles the CLI tool (requires vendor headers to be present)
build:
	@mkdir -p $(BUILD_DIR)
	CGO_ENABLED=1 $(GO) build $(GOFLAGS) -o $(BUILD_DIR)/ollamatokenizer ./cmd/ollamatokenizer

clean:
	rm -rf $(BUILD_DIR)
