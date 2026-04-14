FROM golang:1.25-alpine AS builder

RUN apk add --no-cache build-base curl

WORKDIR /src

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN make build BUILD_DIR=/out

FROM alpine:3.23

RUN apk add --no-cache libstdc++ libgcc
RUN adduser -D -H -u 1000 ollamatokenizer

# Mount the host ollama models directory here (e.g. -v /var/lib/ollama:/ollama-models:ro).
ENV OLLAMA_MODELS=/ollama-models

EXPOSE 11435

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -qO- http://localhost:11435/health || exit 1

COPY --from=builder /out/ollamatokenizer /usr/local/bin/ollamatokenizer

USER ollamatokenizer

ENTRYPOINT ["ollamatokenizer"]
CMD ["serve"]
