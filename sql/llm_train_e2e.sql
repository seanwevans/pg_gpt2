-- End-to-end training regression for a tiny GPT-2 configuration.
SET client_min_messages = warning;
SET extra_float_digits = 3;

TRUNCATE llm_param,
         llm_param_share,
         llm_dataset,
         llm_train_log,
         llm_tensor,
         llm_tensor_rt,
         llm_tensor_map,
         llm_tape;

COPY (SELECT 1) TO PROGRAM 'mkdir -p /mnt/checkpoints';

-- Deterministic helpers for zero and uniform random tensors.
CREATE OR REPLACE FUNCTION llm_tensor_zeros(len int)
RETURNS bytea
LANGUAGE plpgsql
IMMUTABLE
AS $$
DECLARE
    result bytea := '\x'::bytea;
    i int;
BEGIN
    IF len <= 0 THEN
        RETURN '\x'::bytea;
    END IF;

    FOR i IN 1..len LOOP
        result := result || '\x00000000'::bytea;
    END LOOP;

    RETURN result;
END;
$$;

CREATE OR REPLACE FUNCTION llm_tensor_uniform(len int, scale float4 DEFAULT 0.02)
RETURNS bytea
LANGUAGE plpgsql
VOLATILE
AS $$
DECLARE
    result bytea := '\x'::bytea;
    i int;
    val float4;
    be bytea;
BEGIN
    IF len <= 0 THEN
        RETURN '\x'::bytea;
    END IF;

    FOR i IN 1..len LOOP
        val := ((random() - 0.5) * 2 * scale)::float4;
        be := pg_catalog.float4send(val);
        result := result
            || set_byte(
                   set_byte(
                       set_byte(
                           set_byte('\x00000000'::bytea, 0, get_byte(be, 3)),
                           1, get_byte(be, 2)),
                       2, get_byte(be, 1)),
                   3, get_byte(be, 0));
    END LOOP;

    RETURN result;
END;
$$;

CREATE OR REPLACE FUNCTION llm_tensor_fill(len int, value float4)
RETURNS bytea
LANGUAGE plpgsql
IMMUTABLE
AS $$
DECLARE
    result bytea := '\x'::bytea;
    i int;
    be bytea;
BEGIN
    IF len <= 0 THEN
        RETURN '\x'::bytea;
    END IF;

    be := pg_catalog.float4send(value);
    FOR i IN 1..len LOOP
        result := result
            || set_byte(
                   set_byte(
                       set_byte(
                           set_byte('\x00000000'::bytea, 0, get_byte(be, 3)),
                           1, get_byte(be, 2)),
                       2, get_byte(be, 1)),
                   3, get_byte(be, 0));
    END LOOP;

    RETURN result;
END;
$$;

CREATE OR REPLACE PROCEDURE tiny_e2e_prepare(seed float8)
LANGUAGE plpgsql
AS $$
DECLARE
    model TEXT := 'tiny_e2e';
    d INT := 64;
    n_layer INT := 2;
    vocab INT := 32;
    seq_len INT := 8;
    layer INT;
    token_id INT;
BEGIN
    TRUNCATE llm_param,
             llm_dataset,
             llm_train_log,
             llm_tensor,
             llm_tensor_rt,
             llm_tensor_map,
             llm_tape;

    PERFORM setseed(seed);

    -- Token embeddings.
    FOR token_id IN 0..(vocab - 1) LOOP
        INSERT INTO llm_param(model, name, token_id, data, grad, m, v, step)
        VALUES (
            model,
            'wte',
            token_id,
            llm_tensor_uniform(d, 0.02),
            NULL,
            llm_tensor_zeros(d),
            llm_tensor_zeros(d),
            0
        );
    END LOOP;

    -- Positional embeddings for the sequence length used in this test.
    FOR token_id IN 0..(seq_len - 1) LOOP
        INSERT INTO llm_param(model, name, token_id, data, grad, m, v, step)
        VALUES (
            model,
            'wpe',
            token_id,
            llm_tensor_uniform(d, 0.02),
            NULL,
            llm_tensor_zeros(d),
            llm_tensor_zeros(d),
            0
        );
    END LOOP;

    -- Transformer block parameters.
    FOR layer IN 0..(n_layer - 1) LOOP
        INSERT INTO llm_param(model, name, data, grad, m, v, step)
        VALUES
            (model, format('h.%s.ln_1.weight', layer), llm_tensor_fill(d, 1.0), NULL, llm_tensor_zeros(d), llm_tensor_zeros(d), 0),
            (model, format('h.%s.ln_1.bias', layer),   llm_tensor_zeros(d),   NULL, llm_tensor_zeros(d), llm_tensor_zeros(d), 0),
            (model, format('h.%s.attn.c_attn.weight', layer), llm_tensor_uniform(d * 3 * d, 0.02), NULL, llm_tensor_zeros(d * 3 * d), llm_tensor_zeros(d * 3 * d), 0),
            (model, format('h.%s.attn.c_attn.bias', layer),   llm_tensor_zeros(d * 3), NULL, llm_tensor_zeros(d * 3), llm_tensor_zeros(d * 3), 0),
            (model, format('h.%s.attn.c_proj.weight', layer), llm_tensor_uniform(d * d, 0.02), NULL, llm_tensor_zeros(d * d), llm_tensor_zeros(d * d), 0),
            (model, format('h.%s.attn.c_proj.bias', layer),   llm_tensor_zeros(d), NULL, llm_tensor_zeros(d), llm_tensor_zeros(d), 0),
            (model, format('h.%s.mlp.c_fc.weight', layer),    llm_tensor_uniform(d * 4 * d, 0.02), NULL, llm_tensor_zeros(d * 4 * d), llm_tensor_zeros(d * 4 * d), 0),
            (model, format('h.%s.mlp.c_fc.bias', layer),      llm_tensor_zeros(d * 4), NULL, llm_tensor_zeros(d * 4), llm_tensor_zeros(d * 4), 0),
            (model, format('h.%s.mlp.c_proj.weight', layer),  llm_tensor_uniform(d * 4 * d, 0.02), NULL, llm_tensor_zeros(d * 4 * d), llm_tensor_zeros(d * 4 * d), 0),
            (model, format('h.%s.mlp.c_proj.bias', layer),    llm_tensor_zeros(d), NULL, llm_tensor_zeros(d), llm_tensor_zeros(d), 0),
            (model, format('h.%s.ln_2.weight', layer), llm_tensor_fill(d, 1.0), NULL, llm_tensor_zeros(d), llm_tensor_zeros(d), 0),
            (model, format('h.%s.ln_2.bias', layer),   llm_tensor_zeros(d), NULL, llm_tensor_zeros(d), llm_tensor_zeros(d), 0);
    END LOOP;

    -- Final layer norm parameters.
    INSERT INTO llm_param(model, name, data, grad, m, v, step)
    VALUES
        (model, 'ln_f.weight', llm_tensor_fill(d, 1.0), NULL, llm_tensor_zeros(d), llm_tensor_zeros(d), 0),
        (model, 'ln_f.bias',   llm_tensor_zeros(d),     NULL, llm_tensor_zeros(d), llm_tensor_zeros(d), 0);
    -- Synthetic dataset: 100 sequences with deterministic token patterns.
    INSERT INTO llm_dataset(tokens, target)
    SELECT tokens, targets
    FROM (
        SELECT
            ARRAY(
                SELECT ((seq + pos) % 32)
                FROM generate_series(0, 7) AS pos
            )::int[] AS tokens,
            ARRAY(
                SELECT ((seq + pos + 1) % 32)
                FROM generate_series(0, 7) AS pos
            )::int[] AS targets
        FROM generate_series(0, 99) AS seq
    ) AS examples;
END;
$$;

CALL tiny_e2e_prepare(0.123456);

SELECT llm_train('tiny_e2e', 5, 2, 4, 64, 32,
                 dropout_p => 0.0,
                 beta1 => 0.9,
                 beta2 => 0.999,
                 eps => 1e-8,
                 wd => 0.01,
                 lr_max => 0.01,
                 warmup => 2,
                 grad_clip => NULL);

DO $$
DECLARE
    checkpoint_id INT;
    checkpoint_path TEXT;
    original bytea;
    mutated bytea;
    restored bytea;
BEGIN
    PERFORM llm_checkpoint_save('tiny_e2e', 'after 5 steps');

    SELECT id, file_path
      INTO checkpoint_id, checkpoint_path
      FROM llm_checkpoint
     WHERE model = 'tiny_e2e'
     ORDER BY id DESC
     LIMIT 1;

    IF checkpoint_id IS NULL THEN
        RAISE EXCEPTION 'checkpoint metadata missing for model %', 'tiny_e2e';
    END IF;

    PERFORM pg_stat_file(checkpoint_path);

    SELECT data
      INTO original
      FROM llm_param
     WHERE model = 'tiny_e2e'
       AND name = 'wte'
       AND token_id = 0;

    IF original IS NULL THEN
        RAISE EXCEPTION 'expected baseline parameter for model %', 'tiny_e2e';
    END IF;

    UPDATE llm_param
       SET data = llm_tensor_fill(64, 42.0)
     WHERE model = 'tiny_e2e'
       AND name = 'wte'
       AND token_id = 0;

    SELECT data
      INTO mutated
      FROM llm_param
     WHERE model = 'tiny_e2e'
       AND name = 'wte'
       AND token_id = 0;

    IF mutated = original THEN
        RAISE EXCEPTION 'parameter mutation failed to change value';
    END IF;

    PERFORM llm_checkpoint_load('tiny_e2e', checkpoint_id);

    SELECT data
      INTO restored
      FROM llm_param
     WHERE model = 'tiny_e2e'
       AND name = 'wte'
       AND token_id = 0;

    IF restored IS DISTINCT FROM original THEN
        RAISE EXCEPTION 'restored parameter does not match checkpointed value';
    END IF;

    IF (SELECT MAX(step)
          FROM llm_param
         WHERE model = 'tiny_e2e') <> 5 THEN
        RAISE EXCEPTION 'unexpected optimizer step metadata after restore';
    END IF;
END;
$$;

SELECT llm_train('tiny_e2e', 5, 2, 4, 64, 32,
                 dropout_p => 0.0,
                 beta1 => 0.9,
                 beta2 => 0.999,
                 eps => 1e-8,
                 wd => 0.01,
                 lr_max => 0.01,
                 warmup => 2,
                 grad_clip => NULL);

SELECT COUNT(*) AS total_steps,
       MIN(step) AS min_step,
       MAX(step) AS max_step
  FROM llm_train_log
 WHERE model = 'tiny_e2e';

DO $$
DECLARE
    initial_loss FLOAT4;
    final_loss FLOAT4;
    initial_perplexity FLOAT4;
    final_perplexity FLOAT4;
BEGIN
    SELECT loss, exp(loss)
      INTO initial_loss, initial_perplexity
      FROM llm_train_log
     WHERE model = 'tiny_e2e'
     ORDER BY step
     LIMIT 1;

    SELECT loss, exp(loss)
      INTO final_loss, final_perplexity
      FROM llm_train_log
     WHERE model = 'tiny_e2e'
     ORDER BY step DESC
     LIMIT 1;

    IF initial_loss IS NULL OR final_loss IS NULL THEN
        RAISE EXCEPTION 'missing loss records for model %', 'tiny_e2e';
    END IF;

    IF final_loss >= initial_loss - 1e-6 THEN
        RAISE EXCEPTION 'expected loss to decrease (initial=%, final=%)', initial_loss, final_loss;
    END IF;
    IF final_perplexity >= initial_perplexity - 1e-6 THEN
        RAISE EXCEPTION 'expected perplexity to decrease (initial=%, final=%)', initial_perplexity, final_perplexity;
    END IF;
END;
$$;

SELECT 'tiny_e2e perplexity decreased' AS label,
       TRUE AS passed;

CREATE TEMP TABLE first_run_log AS
SELECT step, lr, loss
  FROM llm_train_log
 WHERE model = 'tiny_e2e'
 ORDER BY step;

CALL tiny_e2e_prepare(0.123456);

SELECT llm_train('tiny_e2e', 10, 2, 4, 64, 32,
                 dropout_p => 0.0,
                 beta1 => 0.9,
                 beta2 => 0.999,
                 eps => 1e-8,
                 wd => 0.01,
                 lr_max => 0.01,
                 warmup => 2,
                 grad_clip => NULL);

DO $$
DECLARE
    diff_count INT;
BEGIN
    SELECT COUNT(*)
      INTO diff_count
      FROM (
            SELECT step, lr, loss
              FROM first_run_log
            EXCEPT ALL
            SELECT step, lr, loss
              FROM llm_train_log
             WHERE model = 'tiny_e2e'
           ) AS missing_rows;

    IF diff_count <> 0 THEN
        RAISE EXCEPTION 'reproducibility mismatch: missing % rows', diff_count;
    END IF;

    SELECT COUNT(*)
      INTO diff_count
      FROM (
            SELECT step, lr, loss
              FROM llm_train_log
             WHERE model = 'tiny_e2e'
            EXCEPT ALL
            SELECT step, lr, loss
              FROM first_run_log
           ) AS extra_rows;

    IF diff_count <> 0 THEN
        RAISE EXCEPTION 'reproducibility mismatch: extra % rows', diff_count;
    END IF;
END;
$$;

SELECT 'tiny_e2e loss decreased' AS label,
       COUNT(*) AS logged_steps
  FROM llm_train_log
 WHERE model = 'tiny_e2e';

SELECT 'tiny_e2e reproducibility verified' AS label,
       COUNT(*) AS matching_steps
  FROM first_run_log fr
  JOIN llm_train_log sr
    ON fr.step = sr.step
   AND fr.lr = sr.lr
   AND fr.loss = sr.loss;
