-- Regression test for sequences longer than the legacy 1024-token block size.
SET client_min_messages = warning;

TRUNCATE llm_param,
         llm_dataset,
         llm_train_log,
         llm_tensor,
         llm_tensor_rt,
         llm_tensor_map,
         llm_tape;

-- Minimal parameters for a model with a single embedding dimension.
INSERT INTO llm_param(model, name, token_id, data, grad, m, v, step)
SELECT 'long_ctx', 'wte', token_id, pg_catalog.float4send(0::float4), NULL, NULL, NULL, 0
FROM generate_series(0, 1) AS token_id;

INSERT INTO llm_param(model, name, token_id, data, grad, m, v, step)
SELECT 'long_ctx', 'wpe', position, pg_catalog.float4send(position::float4), NULL, NULL, NULL, 0
FROM generate_series(0, 2047) AS position;

-- Insert a dataset row covering 2048 timesteps.
INSERT INTO llm_dataset(tokens, target)
SELECT
    ARRAY(SELECT 0 FROM generate_series(1, 2048)),
    ARRAY(SELECT 0 FROM generate_series(1, 2048));

SELECT array_length(tokens, 1) AS token_count
FROM llm_dataset
ORDER BY id
LIMIT 1;

SELECT octet_length(llm_embed(tokens, 'long_ctx', 1)) AS embed_bytes
FROM llm_dataset
ORDER BY id
LIMIT 1;

TRUNCATE llm_param,
         llm_dataset,
         llm_train_log,
         llm_tensor,
         llm_tensor_rt,
         llm_tensor_map,
         llm_tape;
