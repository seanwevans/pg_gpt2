# Upgrading pg_gpt2 to pg_gpt3

The repository currently hard-codes GPT-2 conventions across the build, SQL extension, client helpers, and documentation. The bullets below enumerate the concrete edits required to ship a GPT-3 version (`pg_gpt3`). Each item references the exact file locations that must change.

## 1. Rename top-level packaging and docs
- Replace the repository and Docker image identifiers (`pg_gpt2`, `pg-gpt2-demo`, etc.) with `pg_gpt3` across the README header, Docker instructions, and docker-compose assets. 【F:README.md†L1-L55】【F:Dockerfile†L13-L27】
- Update the extension comment in `pg_llm.control` to describe GPT-3 kernels so `CREATE EXTENSION` metadata is accurate. 【F:pg_llm.control†L1-L4】
- Regenerate documentation sections that are branded “pg_gpt2” (developer guide, roadmap, user guide, reproduction playbooks) to reference GPT-3 workflows and filenames. 【F:docs/developer_guide.md†L34-L112】【F:docs/reproducing_gpt2.md†L3-L50】【F:docs/user_guide.md†L1-L20】

## 2. Register GPT-3 model shapes and forward graph
- Extend the `llm_model_config` seed data with GPT-3 families (e.g., 125M, 350M, 1.3B, 175B). GPT-3 uses 2048-token contexts and larger hidden sizes than the current `gpt2-small` row. 【F:sql/pg_llm--0.1.0.sql†L145-L205】
- Fork the GPT-2 specific `llm_forward_gpt2` PL/pgSQL into a GPT-3 variant that reads GPT-3 layer names (e.g., `layers.%d.attn.q_proj`) and dimension ratios. Update every caller (`llm_loss`, `llm_train`, `llm_generate`, etc.) to dispatch to the GPT-3 entry point by default. 【F:sql/pg_llm--0.1.0.sql†L385-L688】【F:sql/llm_block_forward.sql†L1-L214】
- Revisit bias broadcasting assumptions inside `llm_forward_gpt2`—GPT-3 checkpoints store per-channel biases rather than per-token repeats, so the reshape logic must accept 1D arrays instead of `T`-scaled bytea blobs. 【F:sql/llm_block_forward.sql†L85-L143】
- Ensure the causal self-attention kernel continues to scale to 2048-token contexts by verifying the chunking logic in `src/llm_attention.c` and bumping `PG_LLM_ATTENTION_CHUNK` or pack sizes if needed for the larger head dimensions. 【F:src/llm_attention.c†L5-L191】

## 3. Import/export pipeline updates
- Generalize `scripts/convert_gpt2_checkpoint.py` into a GPT-3 aware exporter that understands the Megatron-LM style tensor names used by GPT-3 checkpoints. Update the CLI help text accordingly and adjust the unit tests that validate the gzip stream. 【F:scripts/convert_gpt2_checkpoint.py†L1-L166】【F:tests/test_scripts.py†L14-L190】
- Revise tokenizer ingest and dataset preparation scripts to mention GPT-3 assets and to default to a 2048 block size/overlap so training rows match the wider context. 【F:scripts/ingest_tokenizer.py†L1-L132】【F:scripts/prepare_dataset.py†L4-L203】
- Review `scripts/benchmark_runtime.py` so its model table and prose cover GPT-3 hidden sizes (e.g., 12288-d, 96 heads) instead of GPT-2 presets. 【F:scripts/benchmark_runtime.py†L2-L70】

## 4. Client defaults and tests
- Change every default model identifier in the Python client helpers (`pg_llm_client/generation.py`) from `gpt2-small` to the desired GPT-3 baseline so downstream integrations hit the new configuration automatically. 【F:pg_llm_client/generation.py†L53-L161】
- Adjust regression SQL fixtures/expected outputs to reference GPT-3 model names and to tolerate the larger vocabulary/context if necessary. 【F:expected/llm_train_e2e.out†L1-L200】
- Update test harnesses that import GPT-2 specific scripts or assume 1024-token blocks to align with the GPT-3 tooling and limits. 【F:tests/test_scripts.py†L14-L190】

## 5. Documentation refresh
- Rewrite README sections that walk through GPT-2 reproduction so they instead document GPT-3 fine-tuning, checkpoint conversion, and sampling (including new CLI flags and resource expectations). 【F:README.md†L36-L205】
- Replace the GPT-2 reproduction guide with an equivalent GPT-3 notebook/playbook that covers the larger training recipes, hardware sizing, and hyperparameters. 【F:docs/reproducing_gpt2.md†L3-L50】
- Audit all roadmap and user-guide references so future milestones describe GPT-3-specific enhancements (e.g., tensor parallelism, mixed precision for >100B models). 【F:docs/roadmap.md†L3-L120】

Carrying out the edits above renames the project, aligns the SQL/runtime primitives with GPT-3 parameterizations, and modernizes the Python tooling plus documentation so the extension cleanly targets GPT-3 checkpoints end-to-end.
