# syntax=docker/dockerfile:1

# One-shot provisioning container: ingests the GPT-2 tokenizer and loads model
# weights into a running pg_gpt2 database.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir "psycopg[binary]" numpy

WORKDIR /app
COPY scripts /app/scripts
COPY docker/provision.sh /app/provision.sh
RUN chmod +x /app/provision.sh

ENTRYPOINT ["/app/provision.sh"]
