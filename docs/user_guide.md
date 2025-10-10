# pg_gpt2 User Guide

This guide walks through installing the `pg_llm` PostgreSQL extension, configuring the database, loading pretrained GPT-2 checkpoints, preparing datasets, training and serving models, and managing checkpoints. It concludes with troubleshooting tips and performance tuning guidance tailored to running GPT-2 fully inside PostgreSQL.

## 1. Installation

1. **Install PostgreSQL development headers.** On Debian/Ubuntu systems install `postgresql-server-dev-16` so PGXS can build the extension. If PostgreSQL lives in a custom location, export `PG_CONFIG` so `make` can find it.【F:README.md†L18-L27】
2. **Compile and install the extension.** From the repository root run `make install`. PGXS copies the compiled artifacts into PostgreSQL's extension directory reported by `pg_config --pkglibdir`.【F:README.md†L30-L34】
3. **Activate the extension.** Connect with `psql` and execute `CREATE EXTENSION pg_llm;` in the target database, then verify with `\dx pg_llm` or by querying `pg_extension`.【F:README.md†L30-L34】

## 2. Database Configuration

- **Memory settings.** GPT-2 training and inference use large in-memory tensors. Increase `work_mem` and `maintenance_work_mem` for large models so executor operations stay in memory; consider windowing queries if needed.【F:README.md†L270-L276】
- **Storage layout.** Activations and optimizer scratch space can live in `UNLOGGED` tables to avoid WAL amplification during training workloads.【F:README.md†L272-L277】
- **Core tables.** The extension creates tables such as `llm_param`, `llm_dataset`, `llm_checkpoint`, and tokenizer tables to hold parameters, datasets, checkpoints, and BPE assets.【F:README.md†L63-L73】【F:sql/pg_llm--0.1.0.sql†L133-L175】【F:sql/pg_llm--0.1.0.sql†L557-L618】

## 3. Loading Pretrained Models and Tokenizers

1. **Convert checkpoints.** Use `scripts/convert_gpt2_checkpoint.py` to download or convert a HuggingFace checkpoint into the `.npz` archive consumed by `pg_llm_import_npz`.【F:docs/python_workflow.md†L7-L23】
2. **Import weights.** Load the archive with `SELECT pg_llm_import_npz('/path/to/gpt2-small.npz', 'gpt2-small');` to populate `llm_param`.【F:README.md†L111-L115】【F:sql/pg_llm--0.1.0.sql†L567-L575】
3. **Load tokenizer assets.** Run `scripts/ingest_tokenizer.py` (or the SQL helpers directly) to fill `llm_bpe_vocab` and `llm_bpe_merges`, enabling `llm_encode`/`llm_decode`.【F:docs/python_workflow.md†L25-L41】【F:sql/pg_llm--0.1.0.sql†L603-L631】

## 4. Preparing Datasets

Tokenize raw text into fixed-length sequences with `scripts/prepare_dataset.py`, which writes `(tokens, target)` arrays into the `llm_dataset` table. Specify the DSN, tokenizer, input glob, and block size to match your training configuration.【F:docs/python_workflow.md†L42-L59】【F:sql/pg_llm--0.1.0.sql†L167-L175】

## 5. Training Workflow

1. **Review dataset readiness.** Ensure `llm_dataset` contains rows; `llm_train` raises an exception if the table is empty or becomes empty mid-run.【F:sql/pg_llm--0.1.0.sql†L379-L405】
2. **Launch training.** Call `llm_train(model, n_steps, n_layer, n_head, d_model, vocab_size, ...)` to iterate over randomized dataset batches, run forward/backward passes, apply AdamW updates, and log loss per step.【F:README.md†L124-L143】【F:sql/pg_llm--0.1.0.sql†L229-L313】【F:sql/pg_llm--0.1.0.sql†L356-L410】
3. **Monitor progress.** Training emits `NOTICE` messages with the current step and loss and records metrics in `llm_train_log` for later inspection.【F:sql/pg_llm--0.1.0.sql†L159-L165】【F:sql/pg_llm--0.1.0.sql†L305-L307】【F:sql/pg_llm--0.1.0.sql†L386-L408】

## 6. Inference

1. **Encode prompts.** Use `llm_encode` to convert text into token ids with the loaded BPE tables.【F:README.md†L156-L164】【F:sql/pg_llm--0.1.0.sql†L632-L739】
2. **Generate completions.** `llm_generate(prompt, max_tokens, temperature, topk, topp)` performs autoregressive sampling using the model's logits and decoding pipeline to return text.【F:README.md†L117-L123】【F:README.md†L247-L265】【F:sql/pg_llm--0.1.0.sql†L752-L770】
3. **Decode results.** `llm_decode` maps token ids back to text using the tokenizer tables.【F:README.md†L156-L164】【F:sql/pg_llm--0.1.0.sql†L740-L750】

## 7. Checkpoint Management

- **Export checkpoints.** `llm_checkpoint_save(model, note)` writes a `.npz` snapshot via `pg_llm_export_npz`, records metadata (step, parameter count, file path, note) in `llm_checkpoint`, and stores the archive under `/mnt/checkpoints/`.【F:README.md†L144-L151】【F:sql/pg_llm--0.1.0.sql†L557-L600】
- **Restore checkpoints.** `llm_checkpoint_load(model, checkpoint_id)` resolves the stored file path and re-imports weights via `pg_llm_import_npz`.【F:README.md†L144-L151】【F:sql/pg_llm--0.1.0.sql†L592-L600】
- **List history.** Query `llm_checkpoint` to inspect checkpoints, including creation timestamps and annotations.【F:sql/pg_llm--0.1.0.sql†L557-L565】

## 8. Troubleshooting

- **Build failures (`pg_config` not found).** Ensure the PostgreSQL development package is installed or set `PG_CONFIG` before running `make install`.【F:README.md†L18-L34】
- **Empty training dataset.** `llm_train` errors if `llm_dataset` is empty or drains during training; repopulate the table with `prepare_dataset.py` or additional data.【F:sql/pg_llm--0.1.0.sql†L379-L405】
- **Sequence too long.** The embedding layer enforces GPT-2's 1,024-token context limit and raises an exception when exceeded; adjust block size or truncate input sequences accordingly.【F:sql/pg_llm--0.1.0.sql†L316-L354】
- **Missing embeddings/tokenizer rows.** Errors such as "Missing token embeddings" occur if weights or tokenizer tables were not imported; rerun the import helpers and confirm `llm_param`/`llm_bpe_*` rows exist.【F:sql/pg_llm--0.1.0.sql†L326-L350】【F:sql/pg_llm--0.1.0.sql†L720-L737】
- **Autograd tape empty.** If `llm_train_step` reports an empty tape, ensure `llm_autograd_mode` is populated during the forward pass and that custom SQL does not clear runtime tables prematurely.【F:sql/pg_llm--0.1.0.sql†L272-L313】

## 9. Performance Tuning

- **Adjust memory and batching.** Increase `work_mem`/`maintenance_work_mem` and consider chunked queries so each training step fits the executor's memory context.【F:README.md†L270-L276】
- **Use UNLOGGED scratch tables.** Store large activations or scratch data in `UNLOGGED` tables to reduce WAL pressure.【F:README.md†L272-L277】
- **Leverage optimized kernels.** Core matmul and attention kernels use SIMD-friendly tiling to deliver BLAS-like throughput without extra dependencies, enabling GPT-2 scale workloads entirely inside PostgreSQL.【F:README.md†L270-L275】
- **Parallel housekeeping.** Autograd tape pruning and gradient accumulation can be parallelized safely within transactions when scaling to higher throughput workloads.【F:README.md†L275-L277】

With these steps and tips, you can install `pg_llm`, load pretrained GPT-2 weights, fine-tune on your own datasets, generate text, and maintain checkpoints—all without leaving PostgreSQL.
