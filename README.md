# pg_gpt2

**pg_gpt2** is a complete implementation of the GPT-2 architecture *entirely inside PostgreSQL*.
It extends the database with tensor algebra, automatic differentiation, AdamW optimization, checkpointing, and a Byte-Pair Encoding tokenizer — allowing end-to-end training and text generation purely through SQL and C extensions.

---

## Overview

PostgreSQL is used as both the **storage** and **execution environment** for a large-scale transformer model.
Each layer, weight, and intermediate activation lives in relational tables; tensor operations are implemented as `C` functions returning `BYTEA` buffers.
Every forward pass, gradient computation, and parameter update is a deterministic SQL transaction.

The project demonstrates that a relational database can serve as a full numerical engine, state store, and model runtime — no Python, PyTorch, or external ML stack required.

---

## Prerequisites

Building the extension requires the PostgreSQL server development headers and build tooling so that `pg_config --pgxs` resolves to the `pgxs.mk` makefile. On Debian/Ubuntu systems install the package:

```bash
sudo apt-get install postgresql-server-dev-16
```

If PostgreSQL is installed somewhere custom, set the `PG_CONFIG` environment variable to point at the desired `pg_config` binary before running `make`.

---

## Getting Started

1. **Compile and install the extension.** From the repository root run `make install`. The build uses PGXS and will copy `pg_llm` artifacts into PostgreSQL's extension directory reported by `pg_config --pkglibdir`.
2. **Load the extension in a database.** Connect with `psql` and execute `CREATE EXTENSION pg_llm;` in the target database. This initializes all required tables, functions, and SQL entry points.
3. **Verify availability.** Confirm the extension is active with either `\dx pg_llm` in `psql` or a query such as `SELECT * FROM pg_extension WHERE extname = 'pg_llm';`. Successful output indicates the extension is ready for the workflow described below.

### Docker Image

To simplify evaluation and demos you can run PostgreSQL with the `pg_gpt2` extension pre-installed using the provided Dockerfile.

```bash
# Build the image locally
docker build -t pg-gpt2-demo .

# Start PostgreSQL with pg_llm already installed in the default database
docker run --rm -e POSTGRES_PASSWORD=secret -p 5432:5432 --name pg-gpt2 pg-gpt2-demo
```

The container reuses the official `postgres:16` entrypoint. On first start it creates the default database and automatically enables the `pg_llm` extension so that `psql` connections can immediately run the SQL workflows described below.

---

## Core Design Principles

1. **Postgres as OS** — All computation and persistence live in SQL schemas and C extensions.
2. **Full Reproducibility** — Every step, gradient, and checkpoint is a logged transaction.
3. **Numerical Fidelity** — Bit-level parity with PyTorch’s GPT-2 (`float32`, row-major, GELU, LayerNorm, AdamW).
4. **Composability** — Every tensor op is an SQL function; model architectures are relational graphs.
5. **Auditable Learning** — Because gradients and weights are rows, the entire training process is queryable and replayable.

---

## Architecture Summary

| Component | Description |
|------------|-------------|
| **Tensor Engine** | C implementations of `matmul`, `add`, `gelu`, `softmax`, `layernorm`, `cross_entropy` over contiguous `float32` blobs (`BYTEA`). |
| **Autodiff Engine** | Reverse-mode differentiation recorded in a relational *tape* (`llm_tape`, `llm_tensor_rt`), supporting backpropagation of all GPT-2 ops. |
| **Optimizer** | AdamW with bias correction, decoupled weight decay, gradient clipping, and cosine learning-rate schedule. |
| **Checkpointing** | Import/export weights as `.npz` or `.safetensors` archives. Every snapshot is versioned in `llm_checkpoint`. |
| **Tokenizer** | Native Byte-Pair Encoding (BPE) tokenizer/decoder built from `vocab.json` + `merges.txt`. |
| **Sampling Engine** | Temperature, top-k, and top-p (nucleus) sampling for autoregressive generation. |
| **Training Loop** | SQL functions (`llm_train`, `llm_train_step`, `llm_loss`) orchestrate forward, backward, optimizer updates, and logging. |
| **Inference** | `llm_generate(prompt)` runs encoding → forward → sampling → decoding, returning coherent text completions. |

---

## Key Tables

| Table | Purpose |
|--------|----------|
| `llm_model_config` | Registered model dimensions (layers, heads, embedding size, positions, vocab). |
| `llm_param` | Model parameters, gradients, optimizer state. |
| `llm_dataset` | Tokenized training sequences. |
| `llm_tape` / `llm_tensor_rt` | Computational graph and runtime tensors for autograd. |
| `llm_autograd_mode` | Single-row toggle that signals when forward passes should record autograd tape entries. |
| `llm_checkpoint` | Versioned checkpoint metadata and file paths. |
| `llm_bpe_vocab` / `llm_bpe_merges` | GPT-2 tokenizer vocabulary and merge ranks. |
| `llm_train_log` | Per-step learning rate and loss history. |

---

## Roadmap

See [docs/roadmap.md](docs/roadmap.md) for the upcoming feature roadmap, including GPT-3 style architecture support, mixed-precision execution, and hardware acceleration milestones.

---

## Autograd Workflow

End-to-end training relies on a thin runtime that records every forward op in SQL
so that gradients can be replayed later. The key moving pieces are:

1. **Parameter materialization.** `llm_materialize_params` copies each row in
   `llm_param` into the temporary `llm_tensor` cache and creates a matching row
   in `llm_tensor_rt`. During that copy the helper `pg_llm_autograd_map_param`
   (or its SQL equivalent `INSERT` in the function) must be invoked so the runtime
   tensor id is associated with the original `(model, name, token_id)` tuple. Any
   new C routine that constructs parameter views needs to perform the same mapping
   or gradients will not flow back into `llm_param`. 【F:sql/pg_llm--0.1.0.sql†L403-L438】【F:src/pg_llm_autograd.c†L216-L246】
2. **Forward tape recording.** Every C kernel checks `pg_llm_autograd_enabled()`;
   when the flag is set the inputs and outputs are registered with
   `pg_llm_autograd_track_tensor` and the op is appended to `llm_tape` with any
   metadata (shape, constants, etc.). This produces an ordered tape of all ops in
   the forward pass. 【F:src/pg_llm.c†L19-L210】
3. **Reverse traversal.** `llm_backprop` walks the tape from the newest node back
   to the seed, dispatching gradients based on the recorded `name` field and
   writing results into `llm_tensor_rt.grad`. Once complete, `llm_accumulate_grads`
   copies those buffers back into `llm_param.grad` using the mapping created in
   step 1. 【F:sql/llm_backprop.sql†L1-L78】【F:sql/pg_llm--0.1.0.sql†L439-L456】
4. **Tied embeddings.** GPT-2 reuses the token embedding (`wte`) for the final
   logits projection. After flattening the embedding table into a single matrix
   for `pg_llm_matmul`, ensure that buffer is still mapped to the original
   embedding rows (via `pg_llm_autograd_map_param`) so the logits gradient is
   accumulated back into `wte` rather than a detached copy. 【F:sql/pg_llm--0.1.0.sql†L173-L205】【F:src/pg_llm_autograd.c†L216-L246】

---

## SQL API Reference

### Model Initialization

```sql
SELECT pg_llm_import_npz('/mnt/models/gpt2-small.npz', 'gpt2-small');
```

Imports all pretrained GPT-2 weights into the `llm_param` table.
`llm_model_config` tracks the expected architecture dimensions for each model
and is consulted during import; `gpt2-small` is pre-registered, but custom
models should insert their configuration before calling `pg_llm_import_npz`.

### Forward Pass and Inference

```sql
-- Generate text directly in SQL
SELECT llm_generate('Once upon a time', 80, 0.9, 40, 0.92);

-- Stream tokens as they are produced (step, token_id, token, text, is_complete)
SELECT * FROM llm_generate_stream('Once upon a time', 40, 0.8, 40, 0.95);
```

### Training

```sql
-- Train for 10,000 steps on tokenized text dataset
SELECT llm_train(
  'gpt2-small',
  10000,
  grad_workers => 4,
  prune_workers => 4
);
```

Every step performs:
1. Forward pass → loss (`llm_loss`)
2. Reverse pass (`llm_backprop`)
3. Gradient accumulation
4. AdamW parameter updates
5. Logging to `llm_train_log`

`llm_train` will automatically read the layer count, attention heads, hidden size,
and vocabulary size from `llm_model_config`. Provide overrides for custom
experiments by passing explicit values for `n_layer`, `n_head`, `D`, or `vocab`
when invoking the function.

The training helpers expose knobs for multi-core cleanup work:

- `grad_workers` sets the desired parallel worker count for `llm_accumulate_grads`,
  allowing gradient materialisation from `llm_tensor_rt` into `llm_param` to leverage
  PostgreSQL's parallel query engine.
- `prune_workers` applies the same hinting to `llm_prune_autograd_state`, which clears
  the autograd tape and runtime tensors between steps. Autograd tape pruning is safe
  to parallelise because every runtime tensor row is independent, so this option simply
  tunes planner settings before issuing the deletes.

Both parameters default to `1` (no parallel workers) to preserve existing behaviour.

### Checkpointing

```sql
-- Save a new checkpoint
SELECT llm_checkpoint_save('gpt2-small','after warmup 2k');

-- Restore a checkpoint
SELECT llm_checkpoint_load('gpt2-small',1);
```

### Tokenizer Utilities

```sql
-- Load GPT-2 BPE vocab and merges
SELECT pg_llm_load_bpe_vocab('/mnt/gpt2/vocab.json','gpt2-small');
SELECT pg_llm_load_bpe_merges('/mnt/gpt2/merges.txt','gpt2-small');

-- Encode and decode text
SELECT llm_encode('Hello world!','gpt2-small');
SELECT llm_decode(ARRAY[15496,2159,0],'gpt2-small');
```

### Utility Scripts

The repository includes Python helpers for preparing external assets before
calling the SQL functions above. All scripts live under `scripts/`.

| Script | Purpose |
|--------|---------|
| `convert_gpt2_checkpoint.py` | Download/convert a HuggingFace GPT-2 checkpoint into the gzip-based `.npz` container expected by `pg_llm_import_npz`. |
| `ingest_tokenizer.py` | Load `vocab.json` and `merges.txt` tokenizer assets into `llm_bpe_vocab`/`llm_bpe_merges` using a PostgreSQL connection. |
| `prepare_dataset.py` | Tokenize raw text files with the GPT-2 tokenizer and populate `llm_dataset` with fixed-length `(tokens, target)` arrays. |

Install the optional Python dependencies with:

```
pip install transformers torch psycopg[binary]
```

Examples:

```
# 1. Convert HuggingFace weights to /mnt/models/gpt2-small.npz
python scripts/convert_gpt2_checkpoint.py --source gpt2 --output /mnt/models/gpt2-small.npz

# 2. Load tokenizer assets into PostgreSQL
python scripts/ingest_tokenizer.py \
  --dsn postgresql://postgres@localhost:5432/postgres \
  --model gpt2-small \
  --vocab /mnt/gpt2/vocab.json \
  --merges /mnt/gpt2/merges.txt --truncate

# 3. Tokenize a corpus and fill llm_dataset
python scripts/prepare_dataset.py \
  --dsn postgresql://postgres@localhost:5432/postgres \
  --tokenizer gpt2 \
  --input /mnt/corpus/*.txt \
  --block-size 1024 --truncate
```

An end-to-end walkthrough that stitches the helper scripts together is available
in [docs/python_workflow.md](docs/python_workflow.md), and a fully annotated
Jupyter notebook showing the SQL fine-tuning loop from data ingestion through
generation lives at [docs/fine_tuning_workflow.ipynb](docs/fine_tuning_workflow.ipynb).

---

## Mathematical Fidelity

All core operations follow the official GPT-2 equations:

**Attention**
\[
\mathrm{Attn}(x) = \mathrm{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
\]
with causal masking and learned positional embeddings.

**Feed-Forward**
\[
\mathrm{FFN}(x) = \mathrm{GELU}(xW_1 + b_1)W_2 + b_2
\]

**LayerNorm**
\[
y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}\gamma + \beta
\]

**Loss**
\[
L = -\log \frac{e^{z_t}}{\sum_j e^{z_j}}
\]

**Optimizer (AdamW)**
\[
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat{m}_t &= m_t / (1-\beta_1^t), \quad
\hat{v}_t = v_t / (1-\beta_2^t) \\
\theta_t &= \theta_{t-1} - \eta (\hat{m}_t / (\sqrt{\hat{v}_t}+\epsilon) + \lambda\theta_{t-1})
\end{aligned}
\]

---

## Example: End-to-End Flow

```sql
-- 1. Load model + tokenizer
SELECT pg_llm_import_npz('/mnt/models/gpt2-small.npz','gpt2-small');
SELECT pg_llm_load_bpe_vocab('/mnt/gpt2/vocab.json','gpt2-small');
SELECT pg_llm_load_bpe_merges('/mnt/gpt2/merges.txt','gpt2-small');

-- 2. Encode text
SELECT llm_encode('The database that dreamed of language.','gpt2-small');

-- 3. Generate continuation
SELECT llm_generate('The database that dreamed of language', 40, 0.8, 40, 0.95);

-- 4. Train or fine-tune
SELECT llm_train('gpt2-small', 5000);

-- 5. Save checkpoint
SELECT llm_checkpoint_save('gpt2-small','finetuned on corpus X');
```

---

## Python Client Utilities

Client applications can connect to PostgreSQL using `psycopg` and drive the
text-generation workflow directly from Python. The :mod:`pg_llm_client`
package offers a high-level helper:

```python
import psycopg
from pg_llm_client import PGLLMClient

with psycopg.connect("postgresql://postgres@localhost:5432/postgres") as conn:
    client = PGLLMClient(conn)

    # Single completion with tuned sampling parameters
    print(client.generate("The database that dreamed of language", temperature=0.7))

    # Stream tokens as they arrive
    for event in client.stream("Streaming from SQL", max_tokens=8):
        print(event.text)

    # Retrieve the top beam search candidates
    beams = client.beam_search("Once upon a", beam_width=3, max_tokens=5)
    for beam in beams:
        print(beam.score, beam.text)
```

The helper wraps the SQL API so sampling temperature, beam width, and other
parameters can be adjusted per request without hand-writing SQL in every
client.

---

## Performance Notes

- All tensors are stored as raw `BYTEA` blobs and processed in-memory.
- Core kernels (`pg_llm_matmul`, attention) use a tiled AVX2-aware micro-kernel that falls back to scalar math when SIMD is unavailable, delivering BLAS-class throughput without external dependencies.
- Attention is evaluated in configurable row chunks (default 64 tokens) so that context matrices never exceed a manageable working set, enabling GPT-2 scale sequence lengths inside Postgres.
- For large models, raise `work_mem`/`maintenance_work_mem` and consider chunking your training data via windowed queries so each step fits inside the executor's memory context.
- Store activations and optimizer scratch data in `UNLOGGED` tables (e.g., `CREATE UNLOGGED TABLE llm_activations (...)`) to avoid WAL amplification when materializing large tensors.
- Autograd tape pruning and gradient accumulation can be parallelized safely within a transaction.

---

## Why Do This?

- **Proof of Concept:** show that gradient-based learning can be expressed purely as relational algebra and transaction semantics.
- **Determinism:** every computation is replayable and version-controlled.
- **Integration:** unifies data, model, and training loop under a single ACID engine.
- **Pedagogy:** transparent view into transformer internals, queryable step-by-step.
