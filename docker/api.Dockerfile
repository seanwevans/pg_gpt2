# syntax=docker/dockerfile:1

# HTTP chat endpoint in front of a pg_gpt2 database. The model itself lives in
# PostgreSQL; this image only speaks HTTP.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --no-cache-dir -r /app/server/requirements.txt

COPY server /app/server

# Drop privileges: the endpoint is intended to face the public internet.
RUN useradd --create-home --uid 10001 pggpt2
USER pggpt2

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
