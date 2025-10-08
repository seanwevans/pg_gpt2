#!/usr/bin/env python3
"""Tokenize raw text and populate the llm_dataset table.

This utility reads one or more text files, encodes them with a GPT-2 BPE
(tokenizer loaded via HuggingFace ``transformers``) and writes fixed-length
training examples into ``llm_dataset``.

Usage example:

```
python scripts/prepare_dataset.py \
    --dsn postgresql://postgres@localhost:5432/postgres \
    --tokenizer gpt2 \
    --input ./corpus/*.txt \
    --block-size 1024
```
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

import psycopg


def _execute_values(cur: psycopg.Cursor, query: str, rows: Iterable[Tuple[List[int], List[int]]]) -> int:
    materialized = list(rows)
    if not materialized:
        return 0
    placeholder = "(" + ", ".join(["%s"] * len(materialized[0])) + ")"
    statement = query.replace("VALUES %s", f"VALUES {placeholder}")
    cur.executemany(statement, materialized)
    return len(materialized)


def _lazy_import_tokenizer():
    try:
        from transformers import GPT2TokenizerFast  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "transformers must be installed to prepare the dataset. "
            "Install it with `pip install transformers`."
        ) from exc
    return GPT2TokenizerFast


def iter_input_paths(patterns: Sequence[str]) -> Iterable[Path]:
    for pattern in patterns:
        matched = list(map(Path, glob.glob(pattern)))
        if not matched:
            raise FileNotFoundError(f"No files matched input pattern {pattern!r}")
        for path in matched:
            if not path.is_file():
                continue
            yield path


def load_corpus(patterns: Sequence[str]) -> str:
    texts: List[str] = []
    for path in iter_input_paths(patterns):
        texts.append(path.read_text(encoding="utf-8"))
    return "\n".join(texts)


def chunk_tokens(tokens: Sequence[int], block_size: int) -> Iterator[Tuple[List[int], List[int]]]:
    if block_size < 2:
        raise ValueError("block_size must be at least 2")
    step = block_size
    max_start = len(tokens) - block_size - 1
    for start in range(0, max_start + 1, step):
        window = tokens[start : start + block_size + 1]
        if len(window) < block_size + 1:
            continue
        yield list(window[:-1]), list(window[1:])


def insert_examples(conn: psycopg.Connection, rows: Iterable[Tuple[List[int], List[int]]], batch_size: int = 256) -> int:
    total = 0
    batch: List[Tuple[List[int], List[int]]] = []
    with conn.cursor() as cur:
        for row in rows:
            batch.append(row)
            if len(batch) >= batch_size:
                total += _execute_values(
                    cur, "INSERT INTO llm_dataset(tokens, target) VALUES %s", batch
                )
                batch.clear()
        if batch:
            total += _execute_values(
                cur, "INSERT INTO llm_dataset(tokens, target) VALUES %s", batch
            )
    conn.commit()
    return total


def maybe_truncate(conn: psycopg.Connection, truncate: bool) -> None:
    if not truncate:
        return
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE llm_dataset RESTART IDENTITY")
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", required=True, help="PostgreSQL connection string")
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="HuggingFace tokenizer name or path (e.g., 'gpt2')",
    )
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="Glob patterns of text files to tokenize",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1024,
        help="Sequence length (including next-token target) to generate",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Clear llm_dataset before inserting new examples",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of sequences to insert per batch",
    )

    args = parser.parse_args()

    GPT2TokenizerFast = _lazy_import_tokenizer()
    tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    corpus = load_corpus(args.input)
    token_ids = tokenizer.encode(corpus + tokenizer.eos_token)

    with psycopg.connect(args.dsn) as conn:
        maybe_truncate(conn, args.truncate)
        total = insert_examples(conn, chunk_tokens(token_ids, args.block_size), args.batch_size)

    print(f"Inserted {total} examples into llm_dataset")


if __name__ == "__main__":
    main()
