# syntax=docker/dockerfile:1

FROM postgres:16 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        clang \
        postgresql-server-dev-16 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build/pg_gpt2
COPY . .
RUN make \
    && make install DESTDIR=/tmp/pg_gpt2_install

FROM postgres:16
LABEL org.opencontainers.image.source="https://github.com/postgresml/pg_gpt2" \
      org.opencontainers.image.description="PostgreSQL image bundled with the pg_gpt2 extension for demos and evaluation." \
      org.opencontainers.image.licenses="Apache-2.0"

COPY --from=builder /tmp/pg_gpt2_install/ /

# Initialize the database with the pg_llm extension ready to use.
COPY docker/init_pg_gpt2.sh /docker-entrypoint-initdb.d/pg_gpt2.sh
RUN chmod +x /docker-entrypoint-initdb.d/pg_gpt2.sh

# Expose the default PostgreSQL port
EXPOSE 5432

# Reuse upstream entrypoint / command
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["postgres"]
