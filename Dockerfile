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

# Optionally install CUDA-enabled torch wheels using uv's pip (no system pip needed)
ARG USE_CUDA=false
ENV USE_CUDA=${USE_CUDA}
RUN if [ "$USE_CUDA" = "true" ]; then \
      uv pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio --upgrade --no-cache-dir; \
    fi

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -sf http://127.0.0.1:8000/healthz || exit 1

# For RunPod serverless, use the handler
CMD ["python", "runpod_handler.py"]

