# Requires CGO (for ollama's mlx package glue, which is transitively pulled in by
# the server package even though mlx itself never runs on Linux) and the ollama
# llama-server runner binary on the PATH at runtime.
#
#   make build   # Build the CLI tool
#   make tidy    # go mod tidy
#   make clean   # Remove build artifacts
#
# Set OLLAMA_LIB at runtime if the runner binary is not installed alongside
# this binary (see serve --help).

GO ?= go
GOFLAGS ?=
BUILD_DIR := bin

.PHONY: all build tidy clean

all: build

tidy:
	$(GO) mod tidy $(GOFLAGS)

build:
	@mkdir -p $(BUILD_DIR)
	CGO_ENABLED=1 $(GO) build $(GOFLAGS) -o $(BUILD_DIR)/ollamatokenizer ./cmd/ollamatokenizer

clean:
	rm -rf $(BUILD_DIR)
