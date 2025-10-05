#!/usr/bin/env python3
"""Load GPT-2 tokenizer assets into PostgreSQL.

The llm_bpe_vocab and llm_bpe_merges tables store the Byte Pair Encoding
vocabulary used by GPT-2. This script ingests HuggingFace ``vocab.json`` and
``merges.txt`` files and inserts them using psycopg.

Example usage:

```
python scripts/ingest_tokenizer.py \
    --dsn postgresql://postgres@localhost:5432/postgres \
    --model gpt2-small \
    --vocab ./gpt2/vocab.json \
    --merges ./gpt2/merges.txt
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import psycopg
from psycopg import sql
from psycopg.extras import execute_values


def load_vocab_rows(model: str, path: Path) -> List[Tuple[str, int, str, float | None, bytes]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: List[Tuple[str, int, str, float | None, bytes]] = []
    for token, idx in data.items():
        rows.append((model, int(idx), token, None, token.encode("utf-8")))
    rows.sort(key=lambda r: r[1])
    return rows


def load_merge_rows(model: str, path: Path) -> List[Tuple[str, int, str, str, str]]:
    rows: List[Tuple[str, int, str, str, str]] = []
    rank = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            left, right = line.split()
            rows.append((model, rank, left, right, f"{left} {right}"))
            rank += 1
    return rows


def ingest_vocab(conn: psycopg.Connection, model: str, rows: Iterable[Tuple[str, int, str, float | None, bytes]]) -> None:
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO llm_bpe_vocab(model, token_id, token, score, bytes)
            VALUES %s
            ON CONFLICT (token_id) DO UPDATE
            SET model = EXCLUDED.model,
                token = EXCLUDED.token,
                score = EXCLUDED.score,
                bytes = EXCLUDED.bytes;
            """,
            list(rows),
        )
    conn.commit()


def ingest_merges(conn: psycopg.Connection, model: str, rows: Iterable[Tuple[str, int, str, str, str]]) -> None:
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO llm_bpe_merges(model, rank, left, right, pair)
            VALUES %s
            ON CONFLICT (rank) DO UPDATE
            SET model = EXCLUDED.model,
                left = EXCLUDED.left,
                right = EXCLUDED.right,
                pair = EXCLUDED.pair;
            """,
            list(rows),
        )
    conn.commit()


def maybe_truncate(conn: psycopg.Connection, model: str, truncate: bool) -> None:
    if not truncate:
        return
    with conn.cursor() as cur:
        cur.execute(sql.SQL("DELETE FROM llm_bpe_vocab WHERE model = %s"), (model,))
        cur.execute(sql.SQL("DELETE FROM llm_bpe_merges WHERE model = %s"), (model,))
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", required=True, help="PostgreSQL connection string")
    parser.add_argument("--model", required=True, help="Model name to tag rows with")
    parser.add_argument("--vocab", required=True, type=Path, help="Path to vocab.json")
    parser.add_argument("--merges", required=True, type=Path, help="Path to merges.txt")
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Remove existing vocabulary/merges rows for the model before ingesting",
    )

    args = parser.parse_args()

    with psycopg.connect(args.dsn) as conn:
        maybe_truncate(conn, args.model, args.truncate)
        vocab_rows = load_vocab_rows(args.model, args.vocab)
        merge_rows = load_merge_rows(args.model, args.merges)
        ingest_vocab(conn, args.model, vocab_rows)
        ingest_merges(conn, args.model, merge_rows)

    print(
        f"Loaded {len(vocab_rows)} vocabulary entries and {len(merge_rows)} merges "
        f"for model '{args.model}'."
    )


if __name__ == "__main__":
    main()
