#!/bin/sh
# Load a tokenizer and a set of weights into a running pg_gpt2 database.
#
# Real GPT-2: mount the converted checkpoint and set PG_GPT2_WEIGHTS to its
# path inside the container (see scripts/convert_gpt2_checkpoint.py).
# Otherwise a small randomly-initialised model is created so that every stage
# of the pipeline is exercisable without a 500 MB download.
set -eu

: "${PG_GPT2_DSN:?PG_GPT2_DSN must be set}"
: "${PG_GPT2_MODEL:=gpt2-small}"
: "${PG_GPT2_VOCAB:=/mnt/models/vocab.json}"
: "${PG_GPT2_MERGES:=/mnt/models/merges.txt}"
: "${PG_GPT2_WEIGHTS:=}"

if [ -f "$PG_GPT2_VOCAB" ] && [ -f "$PG_GPT2_MERGES" ]; then
    echo "==> Ingesting tokenizer for '$PG_GPT2_MODEL'"
    python /app/scripts/ingest_tokenizer.py \
        --dsn "$PG_GPT2_DSN" \
        --model "$PG_GPT2_MODEL" \
        --vocab "$PG_GPT2_VOCAB" \
        --merges "$PG_GPT2_MERGES" \
        --truncate
else
    echo "!! No tokenizer at $PG_GPT2_VOCAB / $PG_GPT2_MERGES." >&2
    echo "!! Download GPT-2's vocab.json and merges.txt into ./models first." >&2
    exit 1
fi

if [ -n "$PG_GPT2_WEIGHTS" ]; then
    echo "==> Importing weights from $PG_GPT2_WEIGHTS"
    # pg_llm_import_npz reads the file from the *database* server, so the path
    # must be visible inside the db container.
    psql_sql="SELECT pg_llm_import_npz('$PG_GPT2_WEIGHTS', '$PG_GPT2_MODEL');"
    python - "$PG_GPT2_DSN" "$psql_sql" <<'PY'
import sys
import psycopg

dsn, statement = sys.argv[1], sys.argv[2]
with psycopg.connect(dsn, autocommit=True) as conn:
    conn.execute(statement)
print("Weights imported.")
PY
else
    echo "==> No PG_GPT2_WEIGHTS set; initialising a random model instead"
    python /app/scripts/init_model.py \
        --dsn "$PG_GPT2_DSN" \
        --model "$PG_GPT2_MODEL" \
        --n-layer "${PG_GPT2_N_LAYER:-4}" \
        --n-head "${PG_GPT2_N_HEAD:-4}" \
        --d-model "${PG_GPT2_D_MODEL:-128}" \
        --n-positions "${PG_GPT2_N_POSITIONS:-256}" \
        --vocab "${PG_GPT2_VOCAB_SIZE:-50257}" \
        --replace
fi

echo "==> Provisioning complete."
