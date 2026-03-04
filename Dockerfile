
FROM python:3.11-slim AS base


WORKDIR /app


RUN pip install --no-cache-dir uv


COPY pyproject.toml uv.lock ./


RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    uv sync --no-dev 


COPY src ./src
COPY artifacts ./artifacts

EXPOSE 8000

CMD ["uv", "run","uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS cpu

RUN --mount=type=cache,target=/root/.cache/pip \
    uv pip install --python /app/.venv/bin/python \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision torchaudio

FROM base AS gpu

RUN --mount=type=cache,target=/root/.cache/pip \
    uv pip install --python /app/.venv/bin/python \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio