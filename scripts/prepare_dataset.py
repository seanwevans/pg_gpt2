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


def stream_token_rows(
    tokenizer,
    patterns: Sequence[str],
    block_size: int,
    *,
    read_size: int = 1 << 16,
    overlap: int = 1024,
) -> Iterator[Tuple[List[int], List[int]]]:
    """Yield training rows by tokenizing the corpus without materializing it."""
    if block_size < 2:
        raise ValueError("block_size must be at least 2")

    paths = list(iter_input_paths(patterns))

    token_buffer: List[int] = []
    text_buffer = ""
    overlap_chars = max(overlap, 1)

    eos_token = tokenizer.eos_token or tokenizer.pad_token
    if eos_token is None:  # pragma: no cover - tokenizer should always define one
        raise ValueError("Tokenizer must define an EOS or PAD token")

    def drain_buffer() -> Iterator[Tuple[List[int], List[int]]]:
        while len(token_buffer) >= block_size + 1:
            yield list(token_buffer[:block_size]), list(token_buffer[1 : block_size + 1])
            del token_buffer[:block_size]

    def encode_and_buffer(text: str) -> Iterator[Tuple[List[int], List[int]]]:
        if not text:
            return
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            return
        token_buffer.extend(tokens)
        yield from drain_buffer()

    def flush_text_buffer(flush_all: bool) -> Iterator[Tuple[List[int], List[int]]]:
        nonlocal text_buffer
        if not text_buffer:
            return
        if flush_all:
            segment, text_buffer = text_buffer, ""
        elif len(text_buffer) > overlap_chars:
            cutoff = len(text_buffer) - overlap_chars
            segment = text_buffer[:cutoff]
            text_buffer = text_buffer[cutoff:]
        else:
            return
        yield from encode_and_buffer(segment)

    for index, path in enumerate(paths):
        with path.open("r", encoding="utf-8") as src:
            while True:
                chunk = src.read(read_size)
                if not chunk:
                    break
                text_buffer += chunk
                yield from flush_text_buffer(flush_all=False)

        yield from flush_text_buffer(flush_all=True)

        if index < len(paths) - 1:
            yield from encode_and_buffer("\n")

    yield from encode_and_buffer(eos_token)


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

    with psycopg.connect(args.dsn) as conn:
        maybe_truncate(conn, args.truncate)
        total = insert_examples(
            conn,
            stream_token_rows(tokenizer, args.input, args.block_size),
            args.batch_size,
        )

    print(f"Inserted {total} examples into llm_dataset")


if __name__ == "__main__":
    main()
