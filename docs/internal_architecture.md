# Internal Architecture Reference

This document details the relational layout, autograd tape semantics, and C extension entry points that back the `pg_llm` PostgreSQL extension. It is intended for developers who plan to extend or debug the system.

## Relational Schemas

### Parameter and Optimizer State

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `llm_param` | Persistent storage for all learned tensors along with optimizer buffers. | `(model TEXT, name TEXT, token_id INT)` as the primary key; `data BYTEA`, `grad BYTEA`, `m BYTEA`, `v BYTEA`, `step INT`. |

Biases in the MLP blocks reuse the shared column layout and are mapped into the autograd runtime using `pg_llm_autograd_map_param` when materialized. 【F:sql/pg_llm--0.1.0.sql†L64-L117】【F:sql/pg_llm--0.1.0.sql†L418-L486】

### Runtime Caches for Forward/Backward Passes

| Table | Purpose | Notes |
|-------|---------|-------|
| `llm_tensor` | Materialized tensors that act as a named cache during forward passes. | Created as `UNLOGGED` for speed; columns `id SERIAL`, `name TEXT UNIQUE`, `data BYTEA`, `shape INT[]`, `requires_grad BOOL`. |
| `llm_tensor_rt` | Runtime tensor registry that stores activation buffers and (optionally) gradients captured while autograd is enabled. | Each row has `id SERIAL`, `data BYTEA`, `grad BYTEA`, `shape INT[]`, `requires_grad BOOL`. |
| `llm_tensor_map` | Mapping from persisted parameters to runtime tensor ids so accumulated gradients can flow back into `llm_param`. | Primary key `(model, name, token_id)` referencing `llm_tensor_rt(id)`. |
| `llm_tape` | Ordered list of executed ops in the forward pass. | Columns `id SERIAL`, `name TEXT`, `inputs INT[]`, `output INT`, `extra JSONB`. |
| `llm_autograd_mode` | Single-row guard that toggles whether C kernels should record tape entries. | When empty or set to `false`, runtime kernels skip tape writes. |

The materialization and gradient accumulation helpers (`llm_materialize_params` and `llm_accumulate_grads`) orchestrate moving data between `llm_param`, the caches, and the runtime tensors every training step. 【F:sql/pg_llm--0.1.0.sql†L117-L229】【F:sql/pg_llm--0.1.0.sql†L360-L514】

### Training, Dataset, and Tokenizer Metadata

| Table | Responsibility | Key Fields |
|-------|----------------|------------|
| `llm_dataset` | Tokenized training sequences and shifted targets used for supervised learning. | `tokens INT[]`, `target INT[]` with length checks up to 1024 tokens. |
| `llm_train_log` | Per-step metrics captured during `llm_train`. | `step INT`, `lr FLOAT4`, `loss FLOAT4`. |
| `llm_checkpoint` | Metadata for exported checkpoints stored on disk. | `model TEXT`, `step INT`, `created_at TIMESTAMPTZ`, `file_path TEXT`. |
| `llm_bpe_vocab` | GPT-2 vocabulary tokens, scores, and byte encodings. | `token_id INT PRIMARY KEY`, `token TEXT`, `score FLOAT4`, `bytes BYTEA`. |
| `llm_bpe_merges` | Merge ranks used during byte-pair encoding. | `rank INT PRIMARY KEY`, `left TEXT`, `right TEXT`, `pair TEXT`. |

Tokenizer helpers (`pg_llm_load_bpe_vocab`, `pg_llm_load_bpe_merges`, `llm_encode`, `llm_decode`) use the vocab and merges tables directly for deterministic encoding and decoding. 【F:sql/pg_llm--0.1.0.sql†L229-L360】【F:sql/pg_llm--0.1.0.sql†L514-L760】

## Autograd Tape Semantics

Autograd integrates with PostgreSQL’s SPI API to create a relational recording of the forward graph:

1. **Toggle recording.** Kernels consult `pg_llm_autograd_enabled()`, which checks `llm_autograd_mode`. When `false` no runtime tables are touched. 【F:src/pg_llm_autograd.c†L38-L110】
2. **Tensor tracking.** `pg_llm_autograd_track_tensor` memoizes each `BYTEA *` buffer in a hash table so repeated pointers (e.g., views or shared weights) resolve to the same runtime tensor id. If the tensor is new it is inserted into `llm_tensor_rt` with optional shape metadata. 【F:src/pg_llm_autograd.c†L112-L209】
3. **Tape insertion.** When recording is active, kernels call `pg_llm_autograd_record_tape(name, inputs, n_inputs, output, extra_json)` to append an op row. Inputs reference parent tensor ids returned from `pg_llm_autograd_track_tensor`; `extra` carries JSON metadata such as dimensions or epsilon constants. 【F:src/pg_llm_autograd.c†L64-L145】【F:src/pg_llm_autograd.c†L182-L209】
4. **Parameter mapping.** `pg_llm_autograd_map_param` associates persistent parameters with runtime tensor ids. It infers 1-D shapes when none are provided and upserts entries into `llm_tensor_map`, ensuring gradients can be copied back into `llm_param`. 【F:src/pg_llm_autograd.c†L211-L320】
5. **Backprop traversal.** SQL functions such as `llm_backprop` and `llm_accumulate_grads` walk the tape in reverse order, look up op handlers, and populate `llm_tensor_rt.grad`. After the pass `llm_accumulate_grads` joins through `llm_tensor_map` to update `llm_param.grad`. 【F:sql/llm_backprop.sql†L1-L78】【F:sql/pg_llm--0.1.0.sql†L360-L514】

These primitives allow new kernels to participate in autograd by calling the tracking and tape APIs with consistent tensor ids.

## Extension Function Inventory

C entry points are surfaced as SQL-callable functions. When implementing new ops, use these patterns for argument ordering, dimension metadata, and autograd integration:

| Category | Functions |
|----------|-----------|
| Linear algebra & activations | `pg_llm_matmul`, `pg_llm_add`, `pg_llm_gelu`, `pg_llm_softmax`, `pg_llm_softmax_backward`, `pg_llm_layernorm`, `pg_llm_layernorm_backward`, `pg_llm_transpose`, `pg_llm_dropout`, `pg_llm_dropout_backward`, `pg_llm_ones_like`, `pg_llm_zeros_like`. |
| Attention block | `pg_llm_attention`, `pg_llm_attention_backward`. |
| Losses | `pg_llm_cross_entropy`, `pg_llm_cross_entropy_backward`. |
| Optimizer & schedules | `pg_llm_adamw_step`, `pg_llm_grad_clip`, `pg_llm_lr_schedule`. |
| Checkpointing & tokenizer | `pg_llm_import_npz`, `pg_llm_export_npz`, `pg_llm_load_bpe_vocab`, `pg_llm_load_bpe_merges`. |
| Autograd utilities | `pg_llm_autograd_map_param` (SQL wrapper for parameter mapping). |

Each function is declared in `pg_llm--0.1.0.sql` with `LANGUAGE C` and linked against the compiled extension. The argument lists and return types documented there should be mirrored by any new UDFs you add to ensure compatibility with the PL/pgSQL driver routines like `llm_forward_gpt2`, `llm_train_step`, and `llm_train`. 【F:sql/pg_llm--0.1.0.sql†L1-L360】【F:sql/pg_llm--0.1.0.sql†L360-L514】

## Extending the System

When introducing new operations:

1. Allocate runtime tensors through `pg_llm_autograd_track_tensor`, passing shape and `requires_grad` flags appropriately.
2. Record tape nodes via `pg_llm_autograd_record_tape` using a unique `name` string; ensure the backprop SQL dispatcher can route gradients for that name.
3. If the op produces views over existing parameters (e.g., weight tying or reshaping), call `pg_llm_autograd_map_param` so the backward pass can resolve the original parameter row.
4. Update the SQL orchestration layer (for example `llm_forward_gpt2` or optimizer helpers) to invoke the new kernel and handle autograd mode toggling.

Following these steps keeps the relational tape consistent and guarantees gradients re-integrate with `llm_param` during `llm_accumulate_grads`.
