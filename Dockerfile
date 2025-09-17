FROM nvidia/cuda:12.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app
COPY pyproject.toml /app
RUN /root/.local/bin/uv sync
COPY . /app
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
