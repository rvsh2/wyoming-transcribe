FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 python3-venv \
    libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Install third-party dependencies before copying the source so code changes
# do not invalidate the (torch-sized) dependency layer.
COPY pyproject.toml uv.lock ./
RUN uv venv --python 3.11 /app/.venv && \
    UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124 \
    VIRTUAL_ENV=/app/.venv uv sync --locked --no-dev --no-install-project

COPY README.md ./
COPY cohere_wyoming ./cohere_wyoming
COPY server.py ./server.py
COPY docker-entrypoint.sh ./docker-entrypoint.sh

RUN UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124 \
    VIRTUAL_ENV=/app/.venv uv sync --locked --no-dev && \
    chmod +x /app/docker-entrypoint.sh

ENV PATH="/app/.venv/bin:${PATH}"
ENV VIRTUAL_ENV="/app/.venv"

# 10300 = Wyoming ASR, 8580 = enrollment UI / management API
EXPOSE 10300 8580

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["--uri", "tcp://0.0.0.0:10300", "--language", "pl"]
