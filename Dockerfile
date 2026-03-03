
FROM python:3.11-slim


WORKDIR /app


RUN pip install --no-cache-dir uv


COPY pyproject.toml uv.lock ./


RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    uv sync --no-de
    
COPY ..

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]