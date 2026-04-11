package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
	Use:   "ollamatokenizer",
	Short: "Tokenization server using Ollama's tokenizer implementations",
	Long: `ollamatokenizer provides tokenization that is 1:1 with a running Ollama instance.
It loads only tokenizer metadata from GGUF files — no model weights and no inference.

Set OLLAMA_MODELS to your model directory (e.g. /var/lib/ollama).`,
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
