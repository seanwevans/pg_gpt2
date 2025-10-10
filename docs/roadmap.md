# Roadmap

The following roadmap outlines major enhancements planned for `pg_gpt2`. Timelines are expressed in phases rather than specific dates to accommodate research and experimentation inside PostgreSQL.

## Phase 1 – Broaden Model Coverage

- **GPT-3 style architectures.**
  - Introduce configurable depth/width and attention variants (e.g., parallel residual, dense attention masks).
  - Add activation recomputation and parameter sharing utilities so very deep networks fit within PostgreSQL memory constraints.
  - Expand checkpoint/import tooling to ingest GPT-3 style `.jsonl` weight manifests and tensor sharding formats.
- **Tokenizer flexibility.**
  - Generalize the BPE loader to accept alternate vocabularies (e.g., GPT-3's `encoder.json` / `vocab.bpe`).
  - Provide conversion scripts to retokenize datasets stored in `llm_dataset`.

## Phase 2 – Numerical Efficiency

- **Mixed-precision execution.**
  - Implement `float16`/`bfloat16` tensor kernels with automatic promotion to `float32` for numerically sensitive ops (LayerNorm, softmax).
  - Extend the autograd tape schema to record dtype so gradients propagate without loss.
  - Introduce loss-scaling utilities and guardrails for underflow detection during training.
- **Memory optimizations.**
  - Support on-the-fly tensor compression in `llm_tensor_rt` (chunked storage, delta encoding) to reduce working set size.
  - Add configuration toggles to reuse forward activations across micro-batches.

## Phase 3 – Hardware Acceleration

- **External compute extensions.**
  - Define a pluggable executor API that allows kernels to offload to GPUs via FDWs or background workers.
  - Ship a reference CUDA/OpenCL implementation for matmul/softmax with asynchronous result hydration into Postgres buffers.
- **Vectorized CPU paths.**
  - Provide SIMD-enhanced kernels (AVX-512/NEON) with runtime dispatch based on `pg_config` capabilities.
  - Integrate a microbenchmark suite to validate speedups against the pure SQL/C baselines.

## Phase 4 – Operational Tooling

- **Training orchestration.**
  - Package stored procedures for distributed gradient accumulation across logical replication or foreign tables.
  - Add checkpoint pruning, retention policies, and metrics export (Prometheus view) for monitoring.
- **Model serving.**
  - Build a lightweight HTTP endpoint (via `pg_http` or background worker) that calls `llm_generate` for production inference.

## Phase 5 – Research Extensions

- **Sparse and MoE models.**
  - Experiment with block-sparse attention layouts and router tables stored in relational form.
  - Prototype mixture-of-experts layers using dynamic worker pools inside PostgreSQL.
- **Evaluation suite.**
  - Curate SQL-driven benchmarks (perplexity, throughput, latency) to guide future optimization work.

Feedback and contributions are welcome; open an issue or discussion with the phase and item you are interested in tackling.
