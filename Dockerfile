FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

RUN apt-get update && apt-get install -y \
    cmake \
    curl \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app
COPY pyproject.toml uv.lock /app/
RUN /root/.local/bin/uv sync --locked --no-install-project
RUN /root/.local/bin/uv run --no-sync python -c \
    "import flash_attn, torch; print(flash_attn.__version__, torch.__version__, torch.version.cuda)"
COPY . /app
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
