# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HUGGINGFACE_HUB_CACHE=/cache/hf \
    HF_HOME=/cache/hf \
    TRANSFORMERS_CACHE=/cache/hf \
    UV_LINK_MODE=copy \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

# System deps often needed by torch/transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Ensure the uv-managed virtual environment is used for all runtime commands
ENV PATH="/app/.venv/bin:${PATH}"

# Optionally replace CPU PyTorch with CUDA-enabled wheels (requires host GPU + nvidia runtime)
ARG USE_CUDA=false
ENV USE_CUDA=${USE_CUDA}
RUN /app/.venv/bin/python -m ensurepip --upgrade && \
    /app/.venv/bin/pip install --upgrade pip && \
    if [ "$USE_CUDA" = "true" ]; then \
      /app/.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio --upgrade --no-cache-dir; \
    fi

COPY . .

RUN chmod +x /app/entrypoint.sh

EXPOSE 8000

# Default to API mode. Override MODE for CLI batch/single/eval.
ENV MODE=api

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -sf http://127.0.0.1:8000/healthz || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]

