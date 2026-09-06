#!/usr/bin/env python3
"""Initialise a randomly-weighted GPT-2 style model directly in PostgreSQL.

``pg_llm_import_npz`` is the right tool when you have real pretrained GPT-2
weights.  This script covers the other case: bringing up a *working* instance
(smoke tests, CI, demos, or a chat endpoint you want to click through) without
downloading a checkpoint first.  It registers a model in ``llm_model_config``
and fills ``llm_param`` with weights drawn from ``N(0, std)``, using the same
tensor names and shapes that ``pg_llm_import_npz`` produces.

The generated text is of course meaningless — the point is that every stage of
the pipeline (encode, embed, forward, sample, decode) executes for real.

Example
-------

```
python scripts/init_model.py \
    --dsn postgresql://postgres@localhost:5432/postgres \
    --model gpt2-demo --n-layer 4 --n-head 4 --d-model 128
```
"""

from __future__ import annotations

import argparse
import math

import numpy as np

try:
    import psycopg
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "psycopg is required. Install it with `pip install 'psycopg[binary]'`."
    ) from exc


def _f32(array: np.ndarray) -> bytes:
    """Serialise to the row-major float32 buffer layout pg_llm expects."""

    return np.ascontiguousarray(array, dtype=np.float32).tobytes()


def build_params(
    n_layer: int,
    n_head: int,
    d_model: int,
    n_positions: int,
    vocab: int,
    seed: int,
    std: float,
):
    """Yield ``(name, token_id, buffer)`` rows for every model parameter."""

    rng = np.random.default_rng(seed)

    def normal(*shape: int) -> np.ndarray:
        return rng.normal(0.0, std, size=shape)

    # Token and positional embeddings are stored one row per id so that the
    # embedding lookup in llm_embed is a plain indexed join.
    for token_id in range(vocab):
        yield "wte", token_id, _f32(normal(d_model))
    for position in range(n_positions):
        yield "wpe", position, _f32(normal(d_model))

    # GPT-2 scales the residual projections by 1/sqrt(2 * n_layer).
    residual_std = std / math.sqrt(2.0 * max(n_layer, 1))

    for layer in range(n_layer):
        prefix = f"h.{layer}"
        yield f"{prefix}.ln_1.weight", 0, _f32(np.ones(d_model))
        yield f"{prefix}.ln_1.bias", 0, _f32(np.zeros(d_model))
        yield f"{prefix}.attn.c_attn.weight", 0, _f32(normal(d_model, 3 * d_model))
        yield f"{prefix}.attn.c_attn.bias", 0, _f32(np.zeros(3 * d_model))
        yield f"{prefix}.attn.c_proj.weight", 0, _f32(
            rng.normal(0.0, residual_std, size=(d_model, d_model))
        )
        yield f"{prefix}.attn.c_proj.bias", 0, _f32(np.zeros(d_model))
        yield f"{prefix}.ln_2.weight", 0, _f32(np.ones(d_model))
        yield f"{prefix}.ln_2.bias", 0, _f32(np.zeros(d_model))
        yield f"{prefix}.mlp.c_fc.weight", 0, _f32(normal(d_model, 4 * d_model))
        yield f"{prefix}.mlp.c_fc.bias", 0, _f32(np.zeros(4 * d_model))
        yield f"{prefix}.mlp.c_proj.weight", 0, _f32(
            rng.normal(0.0, residual_std, size=(4 * d_model, d_model))
        )
        yield f"{prefix}.mlp.c_proj.bias", 0, _f32(np.zeros(d_model))

    yield "ln_f.weight", 0, _f32(np.ones(d_model))
    yield "ln_f.bias", 0, _f32(np.zeros(d_model))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", required=True, help="PostgreSQL connection string")
    parser.add_argument("--model", default="gpt2-demo", help="Model name to register")
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-positions", type=int, default=256)
    parser.add_argument(
        "--vocab",
        type=int,
        default=50257,
        help="Vocabulary size; keep this in sync with the ingested tokenizer",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--std",
        type=float,
        default=0.02,
        help="Standard deviation of the normal weight initialisation",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete any existing parameters for this model first",
    )
    args = parser.parse_args()

    if args.d_model % args.n_head:
        raise SystemExit("--d-model must be divisible by --n-head")

    with psycopg.connect(args.dsn) as conn:
        with conn.cursor() as cur:
            if args.replace:
                cur.execute("DELETE FROM llm_param WHERE model = %s", (args.model,))

            cur.execute(
                """
                INSERT INTO llm_model_config
                    (model, n_layer, n_head, d_model, n_positions, vocab)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (model) DO UPDATE SET
                    n_layer = EXCLUDED.n_layer,
                    n_head = EXCLUDED.n_head,
                    d_model = EXCLUDED.d_model,
                    n_positions = EXCLUDED.n_positions,
                    vocab = EXCLUDED.vocab
                """,
                (
                    args.model,
                    args.n_layer,
                    args.n_head,
                    args.d_model,
                    args.n_positions,
                    args.vocab,
                ),
            )

            rows = build_params(
                args.n_layer,
                args.n_head,
                args.d_model,
                args.n_positions,
                args.vocab,
                args.seed,
                args.std,
            )

            count = 0
            with cur.copy(
                "COPY llm_param (model, name, token_id, data, step) FROM STDIN"
            ) as copy:
                for name, token_id, buffer in rows:
                    copy.write_row((args.model, name, token_id, buffer, 0))
                    count += 1
        conn.commit()

    print(f"Initialised {count} parameter rows for model {args.model!r}.")


if __name__ == "__main__":
    main()
