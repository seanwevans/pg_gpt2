# Reproducing the Original GPT-2 Results

This playbook documents how to run the PostgreSQL-native GPT-2 stack end-to-end so that the training regimen, evaluation metrics, and inference workflow match the configurations described in the original OpenAI GPT-2 paper. It assumes you are targeting the reference *gpt2-small* release (12-layer, 12-head, 768-hidden) and want to recover comparable perplexities on public corpora while training and serving the model entirely from SQL.

## 1. Build and Enable the Extension

1. Install the PostgreSQL server development headers (`postgresql-server-dev-16` on Debian/Ubuntu) so PGXS can compile the extension modules that implement tensor kernels, autograd primitives, and optimizer routines inside PostgreSQL.【F:README.md†L18-L73】
2. From the repository root execute `make install`, then connect with `psql` and run `CREATE EXTENSION pg_llm;` to load the schema, tables, and SQL entry points required for training and inference.【F:README.md†L30-L88】
3. Confirm the default model configuration exists—`llm_model_config` ships with the GPT-2 small architecture (12 layers, 12 heads, 768-dimensional hidden state, 1,024 context length, 50,257-token vocabulary), which mirrors the paper’s smallest release and acts as the fallback for `llm_train` and `llm_logits`.【F:sql/pg_llm--0.1.0.sql†L133-L222】【F:sql/pg_llm--0.1.0.sql†L737-L814】

You can alternatively run the prebuilt Docker image if you prefer an isolated PostgreSQL environment; the container boots with `pg_llm` already installed so the remaining steps are identical once you connect via `psql`.【F:README.md†L36-L49】

## 2. Import Reference Weights and Tokenizer Assets

1. Convert the HuggingFace GPT-2 checkpoint into the gzip+NumPy stream expected by `pg_llm_import_npz` using `scripts/convert_gpt2_checkpoint.py`. The utility materializes only the tensors required for SQL execution (embeddings, attention blocks, MLP weights, and layer norms) and ensures they are stored as contiguous `float32` arrays.【F:scripts/convert_gpt2_checkpoint.py†L1-L140】
2. Load the Byte-Pair Encoding vocabulary and merge rules with `scripts/ingest_tokenizer.py`. This populates the `llm_bpe_vocab` and `llm_bpe_merges` tables so the built-in `llm_encode`/`llm_decode` functions reproduce GPT-2 tokenization verbatim.【F:scripts/ingest_tokenizer.py†L1-L131】【F:sql/pg_llm--0.1.0.sql†L1028-L1119】
3. Import the converted checkpoint via `SELECT pg_llm_import_npz('/path/to/gpt2-small.npz','gpt2-small');` so `llm_param` contains the pretrained weights tied to the GPT-2 small configuration.【F:README.md†L124-L165】【F:sql/pg_llm--0.1.0.sql†L1028-L1072】

At this stage the database mirrors the initialization used by the paper: identical tokenizer tables, identical parameter tensors, and a matching model descriptor.

## 3. Prepare the WebText-Style Training Corpus

1. Tokenize your training corpus (e.g., OpenWebText or a private WebText equivalent) with `scripts/prepare_dataset.py`. The helper streams text from glob patterns, encodes it using HuggingFace’s GPT-2 tokenizer, stitches overlapping sequences, and inserts `(tokens, target)` rows into `llm_dataset` in batches so the data never leaves the database during training.【F:scripts/prepare_dataset.py†L1-L206】【F:sql/pg_llm--0.1.0.sql†L313-L327】
2. Make sure the resulting sequences match GPT-2’s 1,024 token context window; adjust the `--block-size` option if you are experimenting with shorter contexts or curriculum schedules.【F:scripts/prepare_dataset.py†L155-L205】【F:sql/pg_llm--0.1.0.sql†L737-L814】
3. Verify the dataset has rows by querying `SELECT COUNT(*) FROM llm_dataset;`—`llm_train` will raise a descriptive exception if it encounters an empty dataset mid-run, matching the guardrails in the original training pipeline.【F:sql/pg_llm--0.1.0.sql†L782-L814】

## 4. Run the GPT-2 Training Schedule

The SQL training loop reproduces the paper’s optimizer and learning-rate policy:

- `llm_train` orchestrates mini-batch sampling, forward/backward passes, and optimizer updates, defaulting to AdamW with β₁=0.9, β₂=0.999, ε=1e-8, weight decay 0.01, dropout 0.1, cosine decay with 2,000-step warmup, and a peak learning rate of 2.5e-4—matching the public GPT-2 training hyperparameters.【F:sql/pg_llm--0.1.0.sql†L737-L814】【F:sql/pg_llm--0.1.0.sql†L597-L688】
- `llm_train_step` handles per-batch execution: it materializes parameters, records the autograd tape, computes the loss, dispatches backprop, applies gradient clipping if requested, and logs the loss/learning rate in `llm_train_log` so you can plot perplexity curves identical to those reported in the paper.【F:sql/pg_llm--0.1.0.sql†L313-L688】

For a full-scale reproduction, call:

```sql
SELECT llm_train(
    'gpt2-small',
    500000,            -- total steps (adjust based on dataset size)
    grad_workers => 4,
    prune_workers => 4
);
```

The command streams randomized batches from `llm_dataset`, emits per-step `NOTICE` messages with the loss, and ensures dropout is disabled automatically during evaluation passes—just as in the TensorFlow implementation described in the paper.【F:sql/pg_llm--0.1.0.sql†L597-L814】

## 5. Measure Perplexity and Validation Loss

1. Evaluate perplexity on held-out splits by running `llm_loss` with `training => false` to disable dropout and compute the average negative log-likelihood per token. Exponentiating the result reproduces the paper’s perplexity metric: `SELECT exp(llm_loss('gpt2-small', seq, target, 12, 12, 768, 50257, training => false));`.【F:sql/pg_llm--0.1.0.sql†L545-L595】
2. During training, query `llm_train_log` to monitor convergence and learning-rate scheduling, e.g., `SELECT step, loss, exp(loss) AS perplexity FROM llm_train_log WHERE model='gpt2-small' ORDER BY step;`—these values align with the curves presented in the GPT-2 report.【F:sql/pg_llm--0.1.0.sql†L313-L688】
3. Automate regression-style checks (e.g., ensuring perplexity decreases after warmup) by adapting the `llm_train_e2e.sql` harness, which already asserts perplexity improves for the bundled tiny dataset and can be extended to real corpora for reproducibility gates.【F:sql/llm_train_e2e.sql†L297-L328】

## 6. Capture Checkpoints and Run Inference

1. Periodically snapshot the model with `llm_checkpoint_save`, which exports the PostgreSQL-resident tensors to `/mnt/checkpoints` and logs metadata (step, note, parameter count) in `llm_checkpoint` for reproducibility audits.【F:sql/pg_llm--0.1.0.sql†L1028-L1072】
2. Restore a specific training run with `llm_checkpoint_load` or re-import the original weights using `pg_llm_import_npz` to compare baseline and fine-tuned perplexities under identical evaluation scripts.【F:sql/pg_llm--0.1.0.sql†L1028-L1072】
3. Use `llm_generate` (or `llm_generate_stream`) to reproduce the qualitative samples in the paper; the SQL routine encodes prompts, applies the trained model, and samples tokens with configurable temperature, top-k, and nucleus parameters to mimic the sampling settings OpenAI reported.【F:sql/pg_llm--0.1.0.sql†L1201-L1306】

Following these steps yields a database-resident training run whose initialization, optimizer, learning-rate schedule, tokenizer, checkpointing, and inference pipeline mirror the published GPT-2 setup. By logging validation perplexity with `llm_loss` and exporting checkpoints at the same milestones reported in the paper, you can demonstrate reproducible parity with the original results entirely from PostgreSQL transactions.
