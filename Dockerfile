FROM python:3.12
RUN apt-get update && apt-get install -y cmake
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
WORKDIR /app
COPY pyproject.toml /app
#COPY protein_search_evals /app/protein_search_evals


RUN /root/.local/bin/uv sync
