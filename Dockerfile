FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    g++ \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

RUN python -m server.data_loader || echo "data prefetch skipped (will fetch at runtime)"

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=80

EXPOSE 80
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=60s \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

CMD ["sh", "-c", "python -m uvicorn server.app:app --host 0.0.0.0 --port ${PORT}"]
