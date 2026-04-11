package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
	Use:   "ollamatokenizer",
	Short: "Tokenization server using Ollama's tokenizer implementations",
	Long: `Tokenization server using Ollama's tokenizer implementations

Set OLLAMA_MODELS to your ollama model directory (e.g. /var/lib/ollama).`,
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
