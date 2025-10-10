# End-to-End Helper Script Workflow

This walkthrough shows how to prepare a PostgreSQL database with pretrained GPT-2
weights, tokenizer assets, and a fine-tuning dataset using the helper scripts in
`scripts/`.

## 1. Convert HuggingFace Weights

Use `convert_gpt2_checkpoint.py` to download a checkpoint from HuggingFace (or
load one from disk) and emit the gzip-compressed archive understood by
`pg_llm_import_npz`.

```bash
python scripts/convert_gpt2_checkpoint.py \
    --source gpt2 \
    --output /mnt/models/gpt2-small.npz
```

The resulting file can be imported into PostgreSQL via:

```sql
SELECT pg_llm_import_npz('/mnt/models/gpt2-small.npz', 'gpt2-small');
```

During import the extension verifies the checkpoint tensors against the
dimensions recorded in `llm_model_config`. The default installation registers
the GPT-2 small architecture; insert a row into that table for custom models
before invoking `pg_llm_import_npz` so the validation can run.

## 2. Ingest Tokenizer Vocabulary

With the weights in place, load the GPT-2 tokenizer vocabulary and merge rules
into the database using `ingest_tokenizer.py`.

```bash
python scripts/ingest_tokenizer.py \
    --dsn postgresql://postgres@localhost:5432/postgres \
    --model gpt2-small \
    --vocab ./gpt2/vocab.json \
    --merges ./gpt2/merges.txt \
    --truncate
```

This populates `llm_bpe_vocab` and `llm_bpe_merges` so SQL functions such as
`llm_encode` and `llm_decode` can translate text to token ids.

## 3. Prepare a Training Dataset

Finally, tokenize a raw text corpus and populate `llm_dataset` using
`prepare_dataset.py`.

```bash
python scripts/prepare_dataset.py \
    --dsn postgresql://postgres@localhost:5432/postgres \
    --tokenizer gpt2 \
    --input ./corpus/*.txt \
    --block-size 1024 \
    --batch-size 512 \
    --truncate
```

The script loads the specified tokenizer with `transformers`, encodes the text
files, and writes fixed-length `(tokens, target)` arrays into PostgreSQL ready
for `llm_train`.

## Putting It Together

1. Convert weights (`convert_gpt2_checkpoint.py`).
2. Load tokenizer assets (`ingest_tokenizer.py`).
3. Populate the dataset table (`prepare_dataset.py`).
4. Import weights and begin training or inference using the SQL functions
   provided by the extension.

Run the helper scripts from the repository root so relative paths to `scripts/`
resolve correctly. Install the optional Python dependencies with:

```bash
pip install transformers torch psycopg[binary]
```

With the tables populated you can now train directly in SQL, for example:

```sql
SELECT llm_train(
    'gpt2-small',
    1000,
    beta1 => 0.9,
    beta2 => 0.999,
    eps => 1e-8,
    wd => 0.01,
    lr_max => 2.5e-4,
    warmup => 2000
);
```

The layer count, head count, hidden size, and vocabulary size are inferred from
`llm_model_config`. Override them by supplying explicit values for the optional
architecture arguments (`n_layer`, `n_head`, `D`, `vocab`) if you are
experimenting with custom checkpoints.

## Client-Side Generation Helpers

Use :mod:`pg_llm_client` for convenience methods that expose temperature
controls, beam search, and streaming token output:

```python
import psycopg
from pg_llm_client import PGLLMClient

with psycopg.connect("postgresql://postgres@localhost:5432/postgres") as conn:
    client = PGLLMClient(conn)
    print(client.generate("Hello from Postgres", temperature=0.6))

    for event in client.stream("Streaming tokens", max_tokens=4):
        print(event.step, event.token, event.text)

    top_candidates = client.beam_search("Beam me", beam_width=2, max_tokens=6)
    for candidate in top_candidates:
        print(candidate.score, candidate.text)
```
