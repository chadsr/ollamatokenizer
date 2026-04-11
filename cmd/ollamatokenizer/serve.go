package main

import (
	"fmt"
	"log"
	"net/http"
	"strings"
	"sync"

	"github.com/chadsr/ollamatokenizer"
	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/spf13/cobra"
)

const defaultPort = 11435

type tokenResponse struct {
	Tokens []int32 `json:"tokens"`
	Count  int     `json:"count"`
}

var tokenCache = struct {
	sync.RWMutex
	m map[string]*ollamatokenizer.Tokenizer
}{m: make(map[string]*ollamatokenizer.Tokenizer)}

func getTokenizer(model string) (*ollamatokenizer.Tokenizer, error) {
	tokenCache.RLock()
	t, ok := tokenCache.m[model]
	tokenCache.RUnlock()
	if ok {
		return t, nil
	}

	tokenCache.Lock()
	defer tokenCache.Unlock()
	if t, ok = tokenCache.m[model]; ok {
		return t, nil
	}

	t, err := ollamatokenizer.New(model)
	if err != nil {
		return nil, err
	}
	tokenCache.m[model] = t
	return t, nil
}

func respondWithTokens(c *gin.Context, tokens []int32, err error) {
	if err != nil {
		if strings.Contains(err.Error(), "not found") {
			c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		} else {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		}
		return
	}
	c.JSON(http.StatusOK, tokenResponse{Tokens: tokens, Count: len(tokens)})
}

func handleTokenizeGenerate(c *gin.Context) {
	var req api.GenerateRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("invalid request: %v", err)})
		return
	}
	if req.Model == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}
	if req.Prompt == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "prompt is required"})
		return
	}

	tok, err := getTokenizer(req.Model)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}

	tokens, err := tok.TokenizeGenerate(req)
	respondWithTokens(c, tokens, err)
}

func handleTokenizeChat(c *gin.Context) {
	var req api.ChatRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("invalid request: %v", err)})
		return
	}
	if req.Model == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}
	if len(req.Messages) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "messages is required"})
		return
	}

	tok, err := getTokenizer(req.Model)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}

	tokens, err := tok.TokenizeChat(req)
	respondWithTokens(c, tokens, err)
}

func handleHealth(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"status": "ok"})
}

var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Start the tokenization HTTP server",
	Long: `Start an HTTP server exposing tokenization endpoints that produce
tokens identical to a running Ollama instance.

Endpoints:
  GET  /health               - health check
  POST /tokenize/generate    - tokenize a prompt (mirrors /api/generate)
  POST /tokenize/chat        - tokenize messages (mirrors /api/chat)

Both tokenization endpoints return: {"tokens": [...], "count": N}`,
	RunE: func(cmd *cobra.Command, args []string) error {
		port, _ := cmd.Flags().GetInt("port")

		gin.SetMode(gin.ReleaseMode)
		r := gin.New()
		r.Use(gin.Logger(), gin.Recovery())

		r.GET("/health", handleHealth)
		r.POST("/tokenize/generate", handleTokenizeGenerate)
		r.POST("/tokenize/chat", handleTokenizeChat)

		addr := fmt.Sprintf(":%d", port)
		log.Printf("ollamatokenizer listening on %s", addr)
		return r.Run(addr)
	},
}

func init() {
	serveCmd.Flags().IntP("port", "p", defaultPort, "port to listen on")
	rootCmd.AddCommand(serveCmd)
}
