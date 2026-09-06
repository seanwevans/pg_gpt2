#!/usr/bin/env python3
"""End-to-end check that a pg_gpt2 database can actually generate text.

Exercises the whole inference path — tokenizer, embedding lookup, transformer
forward pass, sampling and decoding — against a tiny randomly-initialised model
so it needs no checkpoint and no network access.

    python scripts/smoke_test.py --dsn postgresql://postgres@localhost:5432/postgres
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    import psycopg
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "psycopg is required. Install it with `pip install 'psycopg[binary]'`."
    ) from exc

MODEL = "gpt2-smoke"
VOCAB = 256
D_MODEL = 32
N_LAYER = 2
N_HEAD = 2
N_POSITIONS = 64


def load_byte_tokenizer(conn: psycopg.Connection) -> None:
    """Install a byte-level tokenizer with no merges.

    Every GPT-2 base symbol is one byte, so this is a valid (if merge-free)
    tokenizer: encode/decode must still round-trip exactly.
    """

    with conn.cursor() as cur:
        cur.execute("DELETE FROM llm_bpe_vocab WHERE model = %s", (MODEL,))
        cur.execute("DELETE FROM llm_bpe_merges WHERE model = %s", (MODEL,))
        cur.execute(
            """
            INSERT INTO llm_bpe_vocab(model, token_id, token, bytes)
            SELECT %s, e.byte, e.ch, convert_to(e.ch, 'UTF8')
              FROM llm_byte_encoder e
            """,
            (MODEL,),
        )


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "ok" if condition else "FAIL"
    print(f"[{status}] {label}{(' — ' + detail) if detail else ''}")
    if not condition:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", required=True)
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Leave the smoke-test model in the database afterwards",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "init_model.py"),
            "--dsn", args.dsn,
            "--model", MODEL,
            "--n-layer", str(N_LAYER),
            "--n-head", str(N_HEAD),
            "--d-model", str(D_MODEL),
            "--n-positions", str(N_POSITIONS),
            "--vocab", str(VOCAB),
            "--replace",
        ],
        check=True,
    )

    with psycopg.connect(args.dsn, autocommit=True) as conn:
        load_byte_tokenizer(conn)

        with conn.cursor() as cur:
            sample = "Hello, pg_gpt2!"
            cur.execute("SELECT llm_encode(%s::text, %s::text)", (sample, MODEL))
            ids = cur.fetchone()[0]
            check("llm_encode returns tokens", bool(ids), f"{len(ids)} tokens")

            cur.execute("SELECT llm_decode(%s::int[], %s::text)", (ids, MODEL))
            roundtrip = cur.fetchone()[0]
            check("encode/decode round-trips", roundtrip == sample, repr(roundtrip))

            cur.execute(
                "SELECT octet_length(llm_logits(%s::int[], %s::text, last_only => true))",
                (ids, MODEL),
            )
            size = cur.fetchone()[0]
            check(
                "llm_logits returns one row of vocab logits",
                size == VOCAB * 4,
                f"{size} bytes",
            )

            cur.execute(
                "SELECT llm_next_token(%s::int[], %s::text, 1.0::float4, 0, 1.0::float4)",
                (ids, MODEL),
            )
            next_id = cur.fetchone()[0]
            check(
                "llm_next_token samples a valid id",
                isinstance(next_id, int) and 0 <= next_id < VOCAB,
                str(next_id),
            )

            cur.execute(
                """
                SELECT llm_generate(
                    %s::text, %s::int, %s::float4, %s::int, %s::float4, %s::text, %s::int
                )
                """,
                (sample, 8, 0.9, 40, 0.95, MODEL, -1),
            )
            text = cur.fetchone()[0]
            check(
                "llm_generate extends the prompt",
                isinstance(text, str) and text.startswith(sample) and len(text) > len(sample),
                repr(text),
            )

            cur.execute(
                """
                SELECT count(*)
                  FROM llm_generate_stream(
                      %s::text, %s::int, %s::float4, %s::int, %s::float4, %s::text, %s::int
                  )
                """,
                (sample, 5, 0.9, 40, 0.95, MODEL, -1),
            )
            rows = cur.fetchone()[0]
            check("llm_generate_stream yields a row per token", rows == 5, f"{rows} rows")

            # Inference must not leave autograd state behind.
            cur.execute("SELECT count(*) FROM llm_tensor_rt")
            check("inference leaves no autograd tensors", cur.fetchone()[0] == 0)

        if not args.keep:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM llm_param WHERE model = %s", (MODEL,))
                cur.execute("DELETE FROM llm_bpe_vocab WHERE model = %s", (MODEL,))
                cur.execute("DELETE FROM llm_model_config WHERE model = %s", (MODEL,))

    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
