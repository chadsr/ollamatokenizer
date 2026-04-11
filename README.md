# ollamatokenizer

[![CI](https://github.com/chadsr/ollamatokenizer/actions/workflows/ci.yml/badge.svg)](https://github.com/chadsr/ollamatokenizer/actions/workflows/ci.yml)
[![Dependabot Updates](https://github.com/chadsr/ollamatokenizer/actions/workflows/dependabot/dependabot-updates/badge.svg)](https://github.com/chadsr/ollamatokenizer/actions/workflows/dependabot/dependabot-updates)

A simple HTTP server that exposes endpoints for Ollama's built-in tokenization.

## Build

Build the binary:

```shell
make build
```

## Running

```shell
OLLAMA_MODELS=/var/lib/ollama ollamatokenizer serve
```

## Docker

Build and run the container:

```shell
docker build -t ollamatokenizer .
docker run -p 11435:11435 -v /var/lib/ollama:/ollama-models:ro ollamatokenizer
```

The `OLLAMA_MODELS` environment variable defaults to `/ollama-models` inside
the container. Mount your host Ollama model directory at that path.

The server listens on port **11435** by default.

Options:

- `-p`, `--port` -- port to listen on (default: 11435)

## HTTP Endpoints

### GET /health

Health check.

Response:

```json
{
  "status": "ok"
}
```

### POST /tokenize/generate

Tokenize a prompt using the same tokenization as Ollama's `/api/generate`
endpoint. Applies the model's chat template, system prompt, and thinking
settings.

Request body:

```json
{
  "model": "llama3.2:3b",
  "prompt": "Why is the sky blue?",
  "system": "You are a helpful assistant.",
  "think": true
}
```

| Field     | Type    | Required | Description                                              |
|-----------|---------|----------|----------------------------------------------------------|
| `model`   | string  | yes      | Ollama model name (must be pulled)                       |
| `prompt`  | string  | yes      | The user prompt to tokenize                              |
| `system`  | string  | no       | System prompt (overrides the model's default)            |
| `think`   | boolean | no       | Enable thinking/reasoning (auto-detected for thinking models) |

Response:

```json
{
  "tokens": [1, 2998, 338, 278, 6507, 18561, 29973],
  "count": 7
}
```

### POST /tokenize/chat

Tokenize a chat conversation using the same tokenization as Ollama's
`/api/chat` endpoint. Applies the model's chat template, tools, and thinking
settings.

Request body:

```json
{
  "model": "llama3.2:3b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Why is the sky blue?"}
  ],
  "tools": [],
  "think": true
}
```

| Field      | Type    | Required | Description                                               |
|------------|---------|----------|-----------------------------------------------------------|
| `model`    | string  | yes      | Ollama model name (must be pulled)                        |
| `messages` | array   | yes      | List of chat messages with `role` and `content` fields    |
| `tools`    | array   | no       | Tool definitions to include in the prompt                 |
| `think`    | boolean | no       | Enable thinking/reasoning (auto-detected for thinking models) |

Each message object:

| Field      | Type   | Description                        |
|------------|--------|------------------------------------|
| `role`     | string | One of: `system`, `user`, `assistant` |
| `content`  | string | The message text                   |

Response:

```json
{
  "tokens": [1, 2998, 338, 278, 6507, 18561, 29973],
  "count": 7
}
```

### Error Responses

All endpoints return errors in the following format:

```json
{
  "error": "description of the error"
}
```

| HTTP status | Cause                          |
|-------------|--------------------------------|
| 400         | Missing or invalid request body|
| 404         | Model not found                |
| 500         | Internal tokenization error    |
