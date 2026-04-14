# ollamatokenizer

[![CI](https://github.com/chadsr/ollamatokenizer/actions/workflows/ci.yml/badge.svg)](https://github.com/chadsr/ollamatokenizer/actions/workflows/ci.yml)
[![Release](https://github.com/chadsr/ollamatokenizer/actions/workflows/release.yml/badge.svg)](https://github.com/chadsr/ollamatokenizer/actions/workflows/release.yml)
[![Dependabot Updates](https://github.com/chadsr/ollamatokenizer/actions/workflows/dependabot/dependabot-updates/badge.svg)](https://github.com/chadsr/ollamatokenizer/actions/workflows/dependabot/dependabot-updates)

HTTP server exposing Ollama's internal tokenization as API endpoints.

## Build & Run

```shell
make build
OLLAMA_MODELS=/var/lib/ollama ollamatokenizer serve
```

Options: `-p`, `--port` (default: 11435)

### Docker

```shell
docker build -t ollamatokenizer .
docker run -p 11435:11435 -v /var/lib/ollama:/ollama-models:ro ollamatokenizer
```

## Endpoints

### GET /health

Returns `{"status": "ok"}`.

### POST /tokenize

Raw tokenization — no chat template or system prompt.

```json
{"model": "llama3.2:3b", "text": "Why is the sky blue?"}
```

### POST /tokenize/generate

Mirrors `/api/generate`. Applies chat template, system prompt, thinking.

```json
{"model": "llama3.2:3b", "prompt": "Why is the sky blue?", "system": "You are a helpful assistant.", "think": true}
```

### POST /tokenize/chat

Mirrors `/api/chat`. Applies chat template, tools, thinking.

```json
{"model": "llama3.2:3b", "messages": [{"role": "user", "content": "Why is the sky blue?"}], "tools": [], "think": true}
```

## Responses

```json
{"tokens": [1, 2998, 338, 278, 6507, 18561, 29973], "count": 7}
```

```json
{"error": "description"}
```

| Status | Cause                            |
|--------|----------------------------------|
| 400    | Missing or invalid request body, model not found |
| 501    | Unsupported option (suffix, template override, raw mode, context, images) |
| 500    | Tokenization error               |
