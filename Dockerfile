# syntax=docker/dockerfile:1
#
# glibc-based (Ubuntu): libllama.so from the ollama image won't load under musl.
# go.mod carries a leading "v"; the Docker Hub tag does not, so strip it.
ARG OLLAMA_VERSION
FROM ollama/ollama:${OLLAMA_VERSION#v} AS ollama-libs

FROM golang:1.26-bookworm AS builder
ARG OLLAMA_VERSION
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ make curl ca-certificates nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .

# Stage the pinned ollama libs where `make fetch-deps` expects them, then run the
# same target as local dev.
COPY --from=ollama-libs /usr/lib/ollama/libllama.so* /usr/lib/ollama/libggml.so* /usr/lib/ollama/libggml-base.so* /ollama-libs/
RUN make fetch-deps OLLAMA_LIB_DIR=/ollama-libs \
    && CGO_ENABLED=1 go build -trimpath -o /out/ollamatokenizer ./cmd/ollamatokenizer

FROM ubuntu:24.04
RUN apt-get update \
    && apt-get install -y --no-install-recommends libstdc++6 libgcc-s1 wget \
    && rm -rf /var/lib/apt/lists/* \
    && (id ubuntu && userdel -r ubuntu || true) \
    && useradd -l -u 1000 -g nogroup -d /nonexistent -s /usr/sbin/nologin ollamatokenizer

COPY --from=builder /out/ollamatokenizer /usr/local/bin/ollamatokenizer
COPY --from=ollama-libs /usr/lib/ollama/libllama.so* /usr/lib/ollama/libggml.so* /usr/lib/ollama/libggml-base.so* /usr/lib/ollama/
# /usr/lib/ollama isn't on the default ldconfig path; register it.
RUN echo "/usr/lib/ollama" > /etc/ld.so.conf.d/ollama.conf && ldconfig

# Mount host models read-only here: -v /var/lib/ollama:/ollama-models:ro
ENV OLLAMA_MODELS=/ollama-models

EXPOSE 11435
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -qO- http://localhost:11435/health || exit 1

USER ollamatokenizer
ENTRYPOINT ["ollamatokenizer"]
CMD ["serve"]
