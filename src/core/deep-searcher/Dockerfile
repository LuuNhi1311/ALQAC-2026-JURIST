FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

WORKDIR /app

RUN mkdir -p /tmp/uv-cache /app/data /app/logs

COPY pyproject.toml uv.lock LICENSE README.md ./
COPY deepsearcher/ ./deepsearcher/

RUN uv sync 

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

CMD ["uv", "run", "python", "main.py", "--enable-cors", "true"] 