CREATE FUNCTION pg_llm_matmul(a BYTEA, b BYTEA, m INT, k INT, n INT)
RETURNS BYTEA
AS 'MODULE_PATHNAME', 'pg_llm_matmul'
LANGUAGE C STRICT;

CREATE FUNCTION pg_llm_add(a BYTEA, b BYTEA)
RETURNS BYTEA
AS 'MODULE_PATHNAME', 'pg_llm_add'
LANGUAGE C STRICT;

CREATE FUNCTION pg_llm_gelu(a BYTEA)
RETURNS BYTEA
AS 'MODULE_PATHNAME', 'pg_llm_gelu'
LANGUAGE C STRICT;

CREATE FUNCTION pg_llm_softmax(a BYTEA)
RETURNS BYTEA
AS 'MODULE_PATHNAME', 'pg_llm_softmax'
LANGUAGE C STRICT;

CREATE FUNCTION pg_llm_layernorm(x BYTEA, gamma BYTEA, beta BYTEA, eps FLOAT4)
RETURNS BYTEA
AS 'MODULE_PATHNAME', 'pg_llm_layernorm'
LANGUAGE C STRICT;

CREATE FUNCTION pg_llm_cross_entropy(logits BYTEA, target INT)
RETURNS FLOAT4
AS 'MODULE_PATHNAME', 'pg_llm_cross_entropy'
LANGUAGE C STRICT;

CREATE FUNCTION pg_llm_cross_entropy_backward(logits BYTEA, target INT)
RETURNS BYTEA
AS 'MODULE_PATHNAME', 'pg_llm_cross_entropy_backward'
LANGUAGE C STRICT;

CREATE FUNCTION pg_llm_dropout(input BYTEA, p FLOAT4, training BOOLEAN DEFAULT false)
RETURNS BYTEA
AS 'MODULE_PATHNAME', 'pg_llm_dropout'
LANGUAGE C STRICT;

CREATE FUNCTION pg_llm_ones_like(src BYTEA)
RETURNS BYTEA
AS 'MODULE_PATHNAME', 'pg_llm_ones_like'
LANGUAGE C STRICT;

CREATE FUNCTION pg_llm_zeros_like(src BYTEA)
RETURNS BYTEA
AS 'MODULE_PATHNAME', 'pg_llm_zeros_like'
LANGUAGE C STRICT;

CREATE FUNCTION pg_llm_transpose(src BYTEA, rows INT, cols INT)
RETURNS BYTEA
AS 'MODULE_PATHNAME', 'pg_llm_transpose'
LANGUAGE C STRICT;

CREATE FUNCTION pg_llm_dropout_backward(
    input BYTEA,
    output BYTEA,
    grad BYTEA,
    p FLOAT4,
    training BOOLEAN)
RETURNS BYTEA
AS 'MODULE_PATHNAME', 'pg_llm_dropout_backward'
LANGUAGE C STRICT;

CREATE FUNCTION pg_llm_attention(
    x BYTEA,
    w_qkv BYTEA,
    b_qkv BYTEA,
    w_o BYTEA,
    b_o BYTEA,
    n_head INT,
    T INT,
    D INT)
RETURNS BYTEA
AS 'MODULE_PATHNAME', 'pg_llm_attention'
LANGUAGE C STRICT;

CREATE FUNCTION pg_llm_autograd_map_param(
    model TEXT,
    name TEXT,
    token_id INT,
    tensor BYTEA,
    shape INT[] DEFAULT NULL)
RETURNS VOID
AS 'MODULE_PATHNAME', 'pg_llm_autograd_map_param'
LANGUAGE C;
CREATE TYPE attention_grads AS (
    dx BYTEA,
    dw_qkv BYTEA,
    db_qkv BYTEA,
    dw_o BYTEA,
    db_o BYTEA
);

CREATE FUNCTION pg_llm_attention_backward(
    x BYTEA,
    w_qkv BYTEA,
    b_qkv BYTEA,
    w_o BYTEA,
    b_o BYTEA,
    grad_output BYTEA,
    n_head INT,
    T INT,
    D INT)
RETURNS attention_grads
AS 'MODULE_PATHNAME', 'pg_llm_attention_backward'
LANGUAGE C STRICT;

-- AdamW optimizer step
CREATE TYPE adamw_state AS (weight BYTEA, m BYTEA, v BYTEA);

CREATE FUNCTION pg_llm_adamw_step(
    weight BYTEA, grad BYTEA, m BYTEA, v BYTEA,
    lr FLOAT4, beta1 FLOAT4, beta2 FLOAT4, eps FLOAT4,
    weight_decay FLOAT4, step INT)
RETURNS adamw_state
AS 'MODULE_PATHNAME', 'pg_llm_adamw_step'
LANGUAGE C STRICT;

-- Gradient clip
CREATE FUNCTION pg_llm_grad_clip(grad BYTEA, clip FLOAT4)
RETURNS BYTEA
AS 'MODULE_PATHNAME', 'pg_llm_grad_clip'
LANGUAGE C STRICT;

-- LR schedule
CREATE FUNCTION pg_llm_lr_schedule(step INT, warmup INT, total INT, lr_max FLOAT4)
RETURNS FLOAT4
AS 'MODULE_PATHNAME', 'pg_llm_lr_schedule'
LANGUAGE C STRICT;

CREATE TABLE llm_model_config (
    model TEXT PRIMARY KEY,
    n_layer INT NOT NULL CHECK (n_layer > 0),
    n_head INT NOT NULL CHECK (n_head > 0),
    d_model INT NOT NULL CHECK (d_model > 0),
    n_positions INT NOT NULL CHECK (n_positions > 0),
    vocab INT NOT NULL CHECK (vocab > 0),
    CHECK (d_model % n_head = 0)
);

-- Default configuration for the reference GPT-2 small checkpoint
INSERT INTO llm_model_config(model, n_layer, n_head, d_model, n_positions, vocab)
VALUES ('gpt2-small', 12, 12, 768, 1024, 50257)
ON CONFLICT (model) DO NOTHING;

CREATE OR REPLACE FUNCTION llm_get_model_config(p_model TEXT)
RETURNS llm_model_config AS $$
DECLARE
    cfg llm_model_config%ROWTYPE;
BEGIN
    SELECT * INTO cfg
      FROM llm_model_config
     WHERE model = p_model;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Missing llm_model_config entry for model %', p_model
            USING ERRCODE = 'invalid_parameter_value';
    END IF;

    RETURN cfg;
END;
$$ LANGUAGE plpgsql STABLE;

CREATE TABLE llm_param (
    model TEXT,
    name TEXT,
    token_id INT DEFAULT 0,
    data BYTEA,           -- current parameter
    grad BYTEA,           -- accumulated gradient
    m BYTEA,              -- AdamW first moment
    v BYTEA,              -- AdamW second moment
    step INT DEFAULT 0,
    PRIMARY KEY (model, name, token_id)
);

-- Mapping for parameters that share weights with another model.
CREATE TABLE llm_param_share (
    source_model TEXT NOT NULL,
    source_name TEXT NOT NULL,
    source_token_id INT DEFAULT 0 NOT NULL,
    target_model TEXT NOT NULL,
    target_name TEXT NOT NULL,
    target_token_id INT DEFAULT 0 NOT NULL,
    PRIMARY KEY (target_model, target_name, target_token_id)
);

-- Helper view that resolves shared parameters to their underlying storage.
CREATE OR REPLACE VIEW llm_param_resolved AS
    SELECT model, name, token_id, data, grad, m, v, step
      FROM llm_param
    UNION ALL
    SELECT s.target_model AS model,
           s.target_name AS name,
           s.target_token_id AS token_id,
           p.data,
           p.grad,
           p.m,
           p.v,
           p.step
      FROM llm_param_share s
      JOIN llm_param p
        ON p.model = s.source_model
       AND p.name = s.source_name
       AND p.token_id = s.source_token_id;

CREATE OR REPLACE FUNCTION llm_share_param(
    source_model TEXT,
    source_name TEXT,
    source_token_id INT,
    target_model TEXT,
    target_name TEXT DEFAULT NULL,
    target_token_id INT DEFAULT NULL)
RETURNS VOID AS $$
DECLARE
    dest_name TEXT := COALESCE(target_name, source_name);
    dest_token INT := COALESCE(target_token_id, source_token_id);
BEGIN
    PERFORM 1
      FROM llm_param
     WHERE model = source_model
       AND name = source_name
       AND token_id = source_token_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'cannot share parameter %.%[%] because it does not exist',
            source_model, source_name, source_token_id;
    END IF;

    DELETE FROM llm_param
     WHERE model = target_model
       AND name = dest_name
       AND token_id = dest_token;

    INSERT INTO llm_param_share(
        source_model, source_name, source_token_id,
        target_model, target_name, target_token_id)
    VALUES (source_model, source_name, source_token_id,
            target_model, dest_name, dest_token)
    ON CONFLICT (target_model, target_name, target_token_id)
    DO UPDATE SET source_model = EXCLUDED.source_model,
                  source_name = EXCLUDED.source_name,
                  source_token_id = EXCLUDED.source_token_id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION llm_unshare_param(
    target_model TEXT,
    target_name TEXT,
    target_token_id INT DEFAULT 0,
    copy_optimizer_state BOOLEAN DEFAULT true)
RETURNS VOID AS $$
DECLARE
    share_rec RECORD;
BEGIN
    SELECT *
      INTO share_rec
      FROM llm_param_share
     WHERE target_model = llm_unshare_param.target_model
       AND target_name = llm_unshare_param.target_name
       AND target_token_id = llm_unshare_param.target_token_id;

    IF NOT FOUND THEN
        DELETE FROM llm_param_share
         WHERE target_model = llm_unshare_param.target_model
           AND target_name = llm_unshare_param.target_name
           AND target_token_id = llm_unshare_param.target_token_id;
        RETURN;
    END IF;

    DELETE FROM llm_param_share
     WHERE target_model = share_rec.target_model
       AND target_name = share_rec.target_name
       AND target_token_id = share_rec.target_token_id;

    INSERT INTO llm_param(model, name, token_id, data, grad, m, v, step)
    SELECT share_rec.target_model,
           share_rec.target_name,
           share_rec.target_token_id,
           p.data,
           NULL,
           CASE WHEN copy_optimizer_state THEN p.m ELSE NULL::BYTEA END,
           CASE WHEN copy_optimizer_state THEN p.v ELSE NULL::BYTEA END,
           CASE WHEN copy_optimizer_state THEN p.step ELSE 0 END
      FROM llm_param p
     WHERE p.model = share_rec.source_model
       AND p.name = share_rec.source_name
       AND p.token_id = share_rec.source_token_id
    ON CONFLICT (model, name, token_id) DO UPDATE
        SET data = EXCLUDED.data,
            grad = NULL,
            m = EXCLUDED.m,
            v = EXCLUDED.v,
            step = EXCLUDED.step;
END;
$$ LANGUAGE plpgsql;

-- Materialized tensors used during the forward pass (activations, cached weights)
CREATE UNLOGGED TABLE llm_tensor (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE,
    data BYTEA,
    shape INT[],
    requires_grad BOOL DEFAULT false
);

-- Records which model currently owns the shared llm_tensor cache so that the
-- inference path can skip re-materialising unchanged weights.
CREATE UNLOGGED TABLE llm_tensor_owner (
    singleton BOOLEAN PRIMARY KEY DEFAULT true CHECK (singleton),
    model     TEXT NOT NULL
);

-- Single-row toggle to enable/disable autograd recording during forward passes
CREATE UNLOGGED TABLE llm_autograd_mode (
    flag BOOLEAN PRIMARY KEY
);

CREATE TABLE llm_train_log (
    model TEXT,
    step INT,
    lr FLOAT4,
    loss FLOAT4,
    PRIMARY KEY (model, step)
);

CREATE UNLOGGED TABLE llm_dataset (
    id SERIAL PRIMARY KEY,
    tokens INT[],          -- token sequence
    target INT[],          -- shifted targets
    CHECK ((array_length(tokens, 1) IS NULL AND array_length(target, 1) IS NULL)
           OR array_length(tokens, 1) = array_length(target, 1))
);

CREATE OR REPLACE FUNCTION llm_block_forward(
    input BYTEA,
    w_qkv BYTEA,
    b_qkv BYTEA,
    w_o BYTEA,
    b_o BYTEA,
    w_fc BYTEA,
    b_fc BYTEA,
    w_proj BYTEA,
    b_proj BYTEA,
    ln1_g BYTEA,
    ln1_b BYTEA,
    ln2_g BYTEA,
    ln2_b BYTEA,
    n_head INT,
    T INT,
    D INT,
    eps FLOAT4 DEFAULT 1e-5,
    dropout_p FLOAT4 DEFAULT 0.1,
    training BOOLEAN DEFAULT false)
RETURNS BYTEA AS $$
DECLARE
    x BYTEA := input;
    attn BYTEA;
    mlp BYTEA;
    residual2 BYTEA;
BEGIN
    -- 1. LayerNorm
    x := pg_llm_layernorm(x, ln1_g, ln1_b, eps);

    -- 2. Self-Attention
    attn := pg_llm_attention(x, w_qkv, b_qkv, w_o, b_o, n_head, T, D);
    IF training AND dropout_p > 0 THEN
        attn := pg_llm_dropout(attn, dropout_p, training);
    END IF;
    x := pg_llm_add(input, attn);  -- residual 1

    -- 3. LayerNorm
    residual2 := x;
    x := pg_llm_layernorm(x, ln2_g, ln2_b, eps);

    -- 4. Feed-Forward MLP
    mlp := pg_llm_matmul(x, w_fc, T, D, 4*D);
    mlp := pg_llm_add(mlp, b_fc);
    mlp := pg_llm_gelu(mlp);
    mlp := pg_llm_matmul(mlp, w_proj, T, 4*D, D);
    mlp := pg_llm_add(mlp, b_proj);
    IF training AND dropout_p > 0 THEN
        mlp := pg_llm_dropout(mlp, dropout_p, training);
    END IF;

    x := pg_llm_add(residual2, mlp);       -- residual 2
    RETURN x;
END;
$$ LANGUAGE plpgsql STRICT;

CREATE OR REPLACE FUNCTION llm_forward_gpt2(
    input BYTEA,
    model TEXT,
    n_layer INT,
    n_head INT,
    T INT,
    D INT,
    ln_f_weight BYTEA DEFAULT NULL,
    ln_f_bias BYTEA DEFAULT NULL,
    dropout_p FLOAT4 DEFAULT 0.1,
    training BOOLEAN DEFAULT false)
RETURNS BYTEA AS $$
DECLARE
    x BYTEA := input;
    w_qkv BYTEA;
    b_qkv BYTEA;
    w_o BYTEA;
    b_o BYTEA;
    w_fc BYTEA;
    b_fc BYTEA;
    w_proj BYTEA;
    b_proj BYTEA;
    ln1_g BYTEA;
    ln1_b BYTEA;
    ln2_g BYTEA;
    ln2_b BYTEA;
    b_fc_full BYTEA;
    b_proj_full BYTEA;
    expected_fc_bytes INT := T * 4 * D * 4;
    expected_proj_bytes INT := T * D * 4;
    per_token_fc_bytes INT := 4 * D * 4;
    per_token_proj_bytes INT := D * 4;
    final_weight BYTEA := ln_f_weight;
    final_bias BYTEA := ln_f_bias;
BEGIN
    FOR i IN 0..(n_layer-1) LOOP
        SELECT data INTO w_qkv FROM llm_tensor WHERE name = format('h.%s.attn.c_attn.weight', i);
        SELECT data INTO b_qkv FROM llm_tensor WHERE name = format('h.%s.attn.c_attn.bias', i);
        IF b_qkv IS NULL THEN
            SELECT data INTO b_qkv FROM llm_param WHERE name = format('h.%s.attn.c_attn.bias', i) AND token_id = 0 LIMIT 1;
        END IF;
        IF b_qkv IS NULL THEN
            RAISE EXCEPTION 'Missing attention qkv bias for layer %', i;
        END IF;
        SELECT data INTO w_o FROM llm_tensor WHERE name = format('h.%s.attn.c_proj.weight', i);
        SELECT data INTO b_o FROM llm_tensor WHERE name = format('h.%s.attn.c_proj.bias', i);
        IF b_o IS NULL THEN
            SELECT data INTO b_o FROM llm_param WHERE name = format('h.%s.attn.c_proj.bias', i) AND token_id = 0 LIMIT 1;
        END IF;
        IF b_o IS NULL THEN
            RAISE EXCEPTION 'Missing attention proj bias for layer %', i;
        END IF;
        SELECT data INTO w_fc FROM llm_tensor WHERE name = format('h.%s.mlp.c_fc.weight', i);
        SELECT data INTO b_fc FROM llm_tensor WHERE name = format('h.%s.mlp.c_fc.bias', i);
        SELECT data INTO w_proj FROM llm_tensor WHERE name = format('h.%s.mlp.c_proj.weight', i);
        SELECT data INTO b_proj FROM llm_tensor WHERE name = format('h.%s.mlp.c_proj.bias', i);
        SELECT data INTO ln1_g FROM llm_tensor WHERE name = format('h.%s.ln_1.weight', i);
        SELECT data INTO ln1_b FROM llm_tensor WHERE name = format('h.%s.ln_1.bias', i);
        SELECT data INTO ln2_g FROM llm_tensor WHERE name = format('h.%s.ln_2.weight', i);
        SELECT data INTO ln2_b FROM llm_tensor WHERE name = format('h.%s.ln_2.bias', i);

        IF b_fc IS NULL THEN
            RAISE EXCEPTION 'Missing MLP fc bias for layer %', i;
        END IF;
        IF b_proj IS NULL THEN
            RAISE EXCEPTION 'Missing MLP proj bias for layer %', i;
        END IF;

        IF octet_length(b_fc) = expected_fc_bytes THEN
            b_fc_full := b_fc;
        ELSIF octet_length(b_fc) = per_token_fc_bytes THEN
            SELECT string_agg(b_fc, ''::bytea) INTO b_fc_full
            FROM generate_series(1, T);
        ELSE
            RAISE EXCEPTION 'MLP fc bias for layer % has % bytes, expected % (broadcasted) or % (per token)',
                i, octet_length(b_fc), expected_fc_bytes, per_token_fc_bytes;
        END IF;

        IF octet_length(b_proj) = expected_proj_bytes THEN
            b_proj_full := b_proj;
        ELSIF octet_length(b_proj) = per_token_proj_bytes THEN
            SELECT string_agg(b_proj, ''::bytea) INTO b_proj_full
            FROM generate_series(1, T);
        ELSE
            RAISE EXCEPTION 'MLP proj bias for layer % has % bytes, expected % (broadcasted) or % (per token)',
                i, octet_length(b_proj), expected_proj_bytes, per_token_proj_bytes;
        END IF;

        -- The broadcast biases are fresh buffers, so they need their own
        -- runtime ids for gradients to reach llm_param.  Inference never reads
        -- those ids back, and mapping them there would grow llm_tensor_rt by a
        -- row per layer per generated token, so only do this while recording.
        IF EXISTS (SELECT 1 FROM llm_autograd_mode m WHERE m.flag) THEN
            PERFORM pg_llm_autograd_map_param(
                model,
                format('h.%s.mlp.c_fc.bias', i),
                0,
                b_fc_full,
                ARRAY[octet_length(b_fc_full) / 4]
            );

            PERFORM pg_llm_autograd_map_param(
                model,
                format('h.%s.mlp.c_proj.bias', i),
                0,
                b_proj_full,
                ARRAY[octet_length(b_proj_full) / 4]
            );
        END IF;

        x := llm_block_forward(
            x,
            w_qkv,
            b_qkv,
            w_o,
            b_o,
            w_fc,
            b_fc_full,
            w_proj,
            b_proj_full,
            ln1_g,
            ln1_b,
            ln2_g,
            ln2_b,
            n_head, T, D,
            dropout_p => dropout_p,
            training => training);
    END LOOP;
    IF final_weight IS NULL THEN
        SELECT data INTO final_weight
        FROM llm_tensor
        WHERE name = 'ln_f.weight'
        LIMIT 1;
    END IF;

    IF final_weight IS NULL THEN
        SELECT data INTO final_weight
        FROM llm_param
        WHERE name = 'ln_f.weight'
          AND token_id = 0
        LIMIT 1;
    END IF;

    IF final_bias IS NULL THEN
        SELECT data INTO final_bias
        FROM llm_tensor
        WHERE name = 'ln_f.bias'
        LIMIT 1;
    END IF;

    IF final_bias IS NULL THEN
        SELECT data INTO final_bias
        FROM llm_param
        WHERE name = 'ln_f.bias'
          AND token_id = 0
        LIMIT 1;
    END IF;

    IF final_weight IS NULL OR final_bias IS NULL THEN
        RAISE EXCEPTION 'Missing ln_f parameters for final layernorm';
    END IF;

    RETURN pg_llm_layernorm(x, final_weight, final_bias, 1e-5);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION llm_loss(
    model TEXT,
    tokens INT[],
    targets INT[],
    n_layer INT,
    n_head INT,
    D INT,
    vocab INT,
    dropout_p FLOAT4 DEFAULT 0.1,
    training BOOLEAN DEFAULT true)
RETURNS FLOAT4 AS $$
DECLARE
    x BYTEA;
    logits BYTEA;
    loss FLOAT4 := 0.0;
BEGIN
    -- 1. Embed tokens
    x := llm_embed(tokens, model, D);   -- we'll define this below

    -- 2. Forward pass through transformer
    x := llm_forward_gpt2(
        x,
        model,
        n_layer,
        n_head,
        array_length(tokens,1),
        D,
        (SELECT data FROM llm_param_resolved p WHERE p.model = model AND p.name = 'ln_f.weight'),
        (SELECT data FROM llm_param_resolved p WHERE p.model = model AND p.name = 'ln_f.bias'),
        dropout_p => dropout_p,
        training => training);

    -- 3. Final linear projection (tie weights with token_emb).
    --    When this concatenated matrix is handed to the matmul kernel the
    --    runtime id must be registered via pg_llm_autograd_map_param so the
    --    logits gradient is accumulated back into each `wte` row.
    logits := pg_llm_matmul(x,
        (SELECT string_agg(p.data, ''::BYTEA ORDER BY p.token_id)
         FROM llm_param_resolved p
         WHERE p.model = llm_loss.model AND p.name = 'wte'),
        array_length(tokens,1), D, vocab);

    -- 4. Compute loss per token
    FOR i IN 1..array_length(targets,1) LOOP
        loss := loss + pg_llm_cross_entropy(
            substring(logits FROM ((i-1)*vocab+1) FOR vocab)::bytea,
            targets[i]);
    END LOOP;
    RETURN loss / array_length(targets,1);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION llm_train_step(
    model TEXT,
    batch_id INT,
    n_layer INT,
    n_head INT,
    D INT,
    vocab INT,
    dropout_p FLOAT4 DEFAULT 0.1,
    beta1 FLOAT4 DEFAULT 0.9,
    beta2 FLOAT4 DEFAULT 0.999,
    eps FLOAT4 DEFAULT 1e-8,
    wd FLOAT4 DEFAULT 0.01,
    lr_max FLOAT4 DEFAULT 2.5e-4,
    warmup INT DEFAULT 2000,
    total_steps INT DEFAULT 10000,
    grad_clip FLOAT4 DEFAULT NULL,
    grad_workers INT DEFAULT 1,
    prune_workers INT DEFAULT 1)
RETURNS FLOAT4 AS $$
DECLARE
    seq INT[];
    target INT[];
    step_count INT;
    lr FLOAT4;
    loss FLOAT4;
    tape_top INT;
BEGIN
    SELECT tokens, target INTO seq, target
    FROM llm_dataset
    WHERE id = batch_id;

    IF seq IS NULL OR target IS NULL THEN
        RAISE EXCEPTION 'No dataset row with id % for model %', batch_id, model;
    END IF;

    SELECT COALESCE(MAX(step), 0) + 1 INTO step_count
    FROM llm_param_resolved
    WHERE model = llm_train_step.model;

    lr := pg_llm_lr_schedule(step_count, warmup, total_steps, lr_max);
    -- Reset autograd state from the previous step
    PERFORM llm_prune_autograd_state(prune_workers);
    DELETE FROM llm_autograd_mode;

    -- Populate cached tensors for the forward pass
    PERFORM llm_materialize_params(model);

    -- Forward pass with autograd recording enabled
    INSERT INTO llm_autograd_mode(flag) VALUES(true);
    loss := llm_loss(model, seq, target, n_layer, n_head, D, vocab,
                     dropout_p => dropout_p,
                     training => true);
    DELETE FROM llm_autograd_mode;

    SELECT MAX(id) INTO tape_top FROM llm_tape;
    IF tape_top IS NULL THEN
        RAISE EXCEPTION 'Autograd tape empty after forward pass for step %', step_count;
    END IF;

    PERFORM llm_backprop(tape_top, model);
    PERFORM llm_accumulate_grads(model, grad_workers);

    UPDATE llm_param p
    SET (data, m, v, grad, step) = (
        SELECT s.weight, s.m, s.v, NULL::BYTEA, step_count
        FROM pg_llm_adamw_step(
            p.data,
            CASE
                WHEN p.grad IS NULL OR grad_clip IS NULL OR grad_clip <= 0 THEN p.grad
                ELSE pg_llm_grad_clip(p.grad, grad_clip)
            END,
            p.m, p.v,
            lr, beta1, beta2, eps, wd, step_count
        ) AS s
    )
    WHERE p.model = model
       OR EXISTS (
            SELECT 1
              FROM llm_param_share s
             WHERE s.target_model = model
               AND s.source_model = p.model
               AND s.source_name = p.name
               AND s.source_token_id = p.token_id
        );

    INSERT INTO llm_train_log(model, step, lr, loss)
    VALUES(model, step_count, lr, loss);

    -- Free runtime autograd state eagerly so the next step starts clean
    PERFORM llm_prune_autograd_state(prune_workers);

    RETURN loss;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION llm_embed(tokens INT[], p_model TEXT, D INT)
RETURNS BYTEA AS $$
DECLARE
    out BYTEA;
    seq_len INT;
    matched INT;
BEGIN
    seq_len := COALESCE(array_length(tokens, 1), 0);

    IF seq_len = 0 THEN
        RETURN ''::BYTEA;
    END IF;

    -- Flatten summed token and positional embeddings
    SELECT string_agg(
               pg_llm_add(wte.data, wpe.data),
               ''::BYTEA ORDER BY t.ord
           ),
           COUNT(*)
      INTO out, matched
      FROM unnest(tokens) WITH ORDINALITY AS t(token_id, ord)
      JOIN llm_param_resolved wte
        ON wte.model = p_model
       AND wte.name = 'wte'
       AND wte.token_id = t.token_id
      JOIN llm_param_resolved wpe
        ON wpe.model = p_model
       AND wpe.name = 'wpe'
       AND wpe.token_id = t.ord - 1;

    IF out IS NULL THEN
        RAISE EXCEPTION 'Missing token or positional embeddings for model %', p_model;
    END IF;

    IF matched IS DISTINCT FROM seq_len THEN
        RAISE EXCEPTION USING
            ERRCODE = 'data_exception',
            MESSAGE = format('Missing token or positional embeddings for model %s', p_model),
            DETAIL = format('Expected %s embeddings but only found %s', seq_len, matched);
    END IF;

    RETURN out;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION llm_train(
    model TEXT,
    n_steps INT,
    n_layer INT DEFAULT NULL,
    n_head INT DEFAULT NULL,
    D INT DEFAULT NULL,
    vocab INT DEFAULT NULL,
    dropout_p FLOAT4 DEFAULT 0.1,
    beta1 FLOAT4 DEFAULT 0.9,
    beta2 FLOAT4 DEFAULT 0.999,
    eps FLOAT4 DEFAULT 1e-8,
    wd FLOAT4 DEFAULT 0.01,
    lr_max FLOAT4 DEFAULT 2.5e-4,
    warmup INT DEFAULT 2000,
    grad_clip FLOAT4 DEFAULT NULL,
    grad_workers INT DEFAULT 1,
    prune_workers INT DEFAULT 1)
RETURNS VOID AS $$
DECLARE
    loss FLOAT4;
    dataset_ids INT[];
    dataset_size INT;
    idx INT := 1;
    batch_id INT;
    cfg llm_model_config%ROWTYPE;
    effective_n_layer INT := n_layer;
    effective_n_head INT := n_head;
    effective_d INT := D;
    effective_vocab INT := vocab;
BEGIN
    IF effective_n_layer IS NULL OR effective_n_head IS NULL
       OR effective_d IS NULL OR effective_vocab IS NULL THEN
        cfg := llm_get_model_config(model);
        effective_n_layer := COALESCE(effective_n_layer, cfg.n_layer);
        effective_n_head := COALESCE(effective_n_head, cfg.n_head);
        effective_d := COALESCE(effective_d, cfg.d_model);
        effective_vocab := COALESCE(effective_vocab, cfg.vocab);
    END IF;

    IF effective_n_layer IS NULL OR effective_n_head IS NULL
       OR effective_d IS NULL OR effective_vocab IS NULL THEN
        RAISE EXCEPTION 'Missing model configuration parameters for %', model
            USING ERRCODE = 'invalid_parameter_value';
    END IF;

    SELECT array_agg(id ORDER BY random()) INTO dataset_ids FROM llm_dataset;
    dataset_size := COALESCE(array_length(dataset_ids, 1), 0);

    IF dataset_size = 0 THEN
        RAISE EXCEPTION 'llm_dataset is empty; cannot train model %', model;
    END IF;

    FOR i IN 1..n_steps LOOP
        IF idx > dataset_size THEN
            SELECT array_agg(id ORDER BY random()) INTO dataset_ids FROM llm_dataset;
            dataset_size := COALESCE(array_length(dataset_ids, 1), 0);
            IF dataset_size = 0 THEN
                RAISE EXCEPTION 'llm_dataset became empty during training for model %', model;
            END IF;
            idx := 1;
        END IF;

        batch_id := dataset_ids[idx];
        idx := idx + 1;

        loss := llm_train_step(
            model, batch_id,
            effective_n_layer, effective_n_head, effective_d, effective_vocab,
            dropout_p,
            beta1, beta2, eps, wd,
            lr_max, warmup, n_steps,
            grad_clip,
            grad_workers,
            prune_workers);

        RAISE NOTICE 'step %/% loss=%', i, n_steps, loss;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- one node per operation
CREATE UNLOGGED TABLE llm_tape (
    id SERIAL PRIMARY KEY,
    name TEXT,            -- op name: 'matmul','add','gelu','softmax','layernorm'
    inputs INT[],          -- ids of parent tensors
    output INT,            -- id of output tensor
    extra JSONB            -- shape info, constants (e.g., eps, dims)
);

-- NOTE: llm_autograd_mode (the guard flag that toggles autograd recording)
-- is defined once, earlier in this script.

-- store actual data buffers
CREATE UNLOGGED TABLE llm_tensor_rt (
    id SERIAL PRIMARY KEY,
    data BYTEA,
    grad BYTEA,            -- accumulated gradient
    shape INT[],
    requires_grad BOOL DEFAULT false
);

-- Mapping from model parameters to runtime tensor ids for autograd
CREATE UNLOGGED TABLE llm_tensor_map (
    model TEXT,
    name TEXT,
    token_id INT DEFAULT 0 NOT NULL,
    tensor_id INT REFERENCES llm_tensor_rt(id) ON DELETE CASCADE,
    PRIMARY KEY (model, name, token_id)
);

-- NOTE: pg_llm_autograd_map_param is declared once, earlier in this script.

CREATE OR REPLACE FUNCTION llm_materialize_params(p_model TEXT)
RETURNS VOID AS $$
DECLARE
    rec RECORD;
    tensor_name TEXT;
    tensor_id INT;
BEGIN
    -- Clear cached tensors for this step
    DELETE FROM llm_tensor;
    DELETE FROM llm_tensor_owner;
    DELETE FROM llm_tensor_map WHERE model = p_model;

    -- Copy parameters into the tensor cache and create runtime tensors
    FOR rec IN
        SELECT name, token_id, data
        FROM (
            SELECT p.name, p.token_id, p.data
            FROM llm_param p
            WHERE p.model = p_model
            UNION ALL
            SELECT s.target_name AS name,
                   s.target_token_id AS token_id,
                   src.data
            FROM llm_param_share s
            JOIN llm_param src
              ON src.model = s.source_model
             AND src.name = s.source_name
             AND src.token_id = s.source_token_id
            WHERE s.target_model = p_model
        ) params
    LOOP
        tensor_name := CASE
                           WHEN rec.token_id = 0 THEN rec.name
                           ELSE format('%s.%s', rec.name, rec.token_id)
                       END;

        INSERT INTO llm_tensor(name, data, requires_grad)
        VALUES (tensor_name, rec.data, true)
        ON CONFLICT (name) DO UPDATE
            SET data = EXCLUDED.data,
                requires_grad = EXCLUDED.requires_grad;

        IF rec.name LIKE 'h.%' || '.mlp.c_fc.bias'
           OR rec.name LIKE 'h.%' || '.mlp.c_proj.bias' THEN
            PERFORM pg_llm_autograd_map_param(
                p_model,
                rec.name,
                rec.token_id,
                rec.data,
                ARRAY[octet_length(rec.data) / 4]
            );
        END IF;

        INSERT INTO llm_tensor_rt(data, grad, shape, requires_grad)
        VALUES (rec.data, NULL, NULL, true)
        RETURNING id INTO tensor_id;

        INSERT INTO llm_tensor_map(model, name, token_id, tensor_id)
        VALUES (p_model, rec.name, rec.token_id, tensor_id)
        ON CONFLICT (model, name, token_id) DO UPDATE
            SET tensor_id = EXCLUDED.tensor_id;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION llm_accumulate_grads(p_model TEXT, p_workers INT DEFAULT 1)
RETURNS VOID AS $$
DECLARE
    effective_workers INT := GREATEST(COALESCE(p_workers, 1), 1);
BEGIN
    IF effective_workers > 1 THEN
        PERFORM set_config('max_parallel_workers_per_gather', effective_workers::TEXT, true);
        PERFORM set_config('parallel_setup_cost', '0', true);
        PERFORM set_config('parallel_tuple_cost', '0', true);
    END IF;

    -- Clear stale gradients for this model
    UPDATE llm_param p
    SET grad = NULL
    WHERE p.model = p_model
       OR EXISTS (
            SELECT 1
              FROM llm_param_share s
             WHERE s.target_model = p_model
               AND s.source_model = p.model
               AND s.source_name = p.name
               AND s.source_token_id = p.token_id
        );

    -- Populate grads from runtime tensors recorded during autograd
    UPDATE llm_param p
    SET grad = t.grad
    FROM llm_tensor_map m
    JOIN llm_tensor_rt t ON t.id = m.tensor_id
    WHERE p.model = p_model
      AND p.model = m.model
      AND p.name = m.name
      AND p.token_id = m.token_id
      AND t.grad IS NOT NULL;

    UPDATE llm_param p
    SET grad = t.grad
    FROM llm_param_share s
    JOIN llm_tensor_map m
      ON m.model = s.target_model
     AND m.name = s.target_name
     AND m.token_id = s.target_token_id
    JOIN llm_tensor_rt t ON t.id = m.tensor_id
    WHERE s.target_model = p_model
      AND p.model = s.source_model
      AND p.name = s.source_name
      AND p.token_id = s.source_token_id
      AND t.grad IS NOT NULL;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION llm_prune_autograd_state(p_workers INT DEFAULT 1)
RETURNS VOID AS $$
DECLARE
    effective_workers INT := GREATEST(COALESCE(p_workers, 1), 1);
BEGIN
    IF effective_workers > 1 THEN
        PERFORM set_config('max_parallel_workers_per_gather', effective_workers::TEXT, true);
        PERFORM set_config('parallel_setup_cost', '0', true);
        PERFORM set_config('parallel_tuple_cost', '0', true);
    END IF;

    DELETE FROM llm_tape;
    DELETE FROM llm_tensor_rt;
END;
$$ LANGUAGE plpgsql;

-- Reference sketch of a single training step. This is illustrative
-- pseudo-code (it uses PL/pgSQL PERFORM outside a function body and
-- placeholder identifiers), so it is kept commented out; executing it as
-- part of CREATE EXTENSION would fail. See llm_train_step()/llm_train()
-- below for the real implementation.
-- BEGIN;
--   PERFORM llm_prune_autograd_state();
--
--   -- Forward pass with autograd enabled
--   INSERT INTO llm_autograd_mode VALUES(true);
--   PERFORM llm_loss(
--       'gpt2-small',
--       seq,
--       target,
--       (SELECT n_layer FROM llm_model_config WHERE model = 'gpt2-small'),
--       (SELECT n_head FROM llm_model_config WHERE model = 'gpt2-small'),
--       (SELECT d_model FROM llm_model_config WHERE model = 'gpt2-small'),
--       (SELECT vocab FROM llm_model_config WHERE model = 'gpt2-small'));
--
--   -- Reverse pass
--   PERFORM llm_backprop((SELECT MAX(id) FROM llm_tape), 'gpt2-small');
--
--   -- Gradient accumulation
--   PERFORM llm_accumulate_grads('gpt2-small');
--
--   -- Optimizer update
--   PERFORM llm_train_step(...);
--
-- COMMIT;

CREATE FUNCTION pg_llm_softmax_backward(y BYTEA, dy BYTEA)
RETURNS BYTEA
AS 'MODULE_PATHNAME', 'pg_llm_softmax_backward'
LANGUAGE C STRICT;

CREATE TYPE layernorm_grads AS (dx BYTEA, dgamma BYTEA, dbeta BYTEA);

CREATE FUNCTION pg_llm_layernorm_backward(
    x BYTEA, dy BYTEA, gamma BYTEA, eps FLOAT4)
RETURNS layernorm_grads
AS 'MODULE_PATHNAME', 'pg_llm_layernorm_backward'
LANGUAGE C STRICT;

CREATE TABLE llm_checkpoint (
    id           SERIAL PRIMARY KEY,
    model        TEXT,
    step         INT,
    created_at   TIMESTAMPTZ DEFAULT now(),
    note         TEXT,
    n_params     BIGINT,
    file_path    TEXT
);

CREATE FUNCTION pg_llm_import_npz(path TEXT, model TEXT)
RETURNS void
AS 'MODULE_PATHNAME', 'pg_llm_import_npz'
LANGUAGE C STRICT;

CREATE FUNCTION pg_llm_export_npz(path TEXT, model TEXT)
RETURNS void
AS 'MODULE_PATHNAME', 'pg_llm_export_npz'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION llm_checkpoint_save(model TEXT, note TEXT)
RETURNS VOID AS $$
DECLARE
    path TEXT := format('/mnt/checkpoints/%s-step%s.npz', model,
                        (SELECT MAX(step) FROM llm_param_resolved WHERE model=model));
    n BIGINT;
BEGIN
    PERFORM pg_llm_export_npz(path, model);
    SELECT COUNT(*) INTO n FROM llm_param_resolved WHERE model=model;
    INSERT INTO llm_checkpoint(model,step,n_params,file_path,note)
    VALUES(model,(SELECT MAX(step) FROM llm_param_resolved WHERE model=model),
           n,path,note);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION llm_checkpoint_load(model TEXT, checkpoint_id INT)
RETURNS VOID AS $$
DECLARE
    path TEXT;
BEGIN
    SELECT file_path INTO path FROM llm_checkpoint
    WHERE id=checkpoint_id;
    PERFORM pg_llm_import_npz(path, model);
END;
$$ LANGUAGE plpgsql;

CREATE TABLE llm_bpe_vocab (
    model TEXT NOT NULL,
    token_id INT NOT NULL,
    token TEXT,               -- raw text form
    score FLOAT4,             -- optional rank / freq
    bytes BYTEA,              -- UTF-8 representation
    -- Keyed by model as well as id so that one database can hold the
    -- tokenizers for several models at once.
    PRIMARY KEY (model, token_id)
);

CREATE INDEX llm_bpe_vocab_token_idx ON llm_bpe_vocab (model, token);

CREATE TABLE llm_bpe_merges (
    model TEXT NOT NULL,
    rank INT NOT NULL,
    "left" TEXT,
    "right" TEXT,
    pair TEXT,                -- "left right"
    PRIMARY KEY (model, rank)
);

CREATE INDEX llm_bpe_merges_pair_idx ON llm_bpe_merges (model, "left", "right");

CREATE FUNCTION pg_llm_load_bpe_vocab(path TEXT, model TEXT)
RETURNS void
AS 'MODULE_PATHNAME', 'pg_llm_load_bpe_vocab'
LANGUAGE C STRICT;

CREATE FUNCTION pg_llm_load_bpe_merges(path TEXT, model TEXT)
RETURNS void
AS 'MODULE_PATHNAME', 'pg_llm_load_bpe_merges'
LANGUAGE C STRICT;

-- Tokenizer assets are loaded after installation, once vocab.json/merges.txt are
-- available on the server filesystem, e.g.:
--   SELECT pg_llm_load_bpe_vocab('/mnt/gpt2/vocab.json','gpt2-small');
--   SELECT pg_llm_load_bpe_merges('/mnt/gpt2/merges.txt','gpt2-small');

-- GPT-2 works on bytes, not characters: every input byte is first mapped to a
-- printable code point (OpenAI's bytes_to_unicode table) so that the BPE merge
-- table can be expressed as ordinary text.  This table is that mapping.
CREATE TABLE llm_byte_encoder (
    byte INT PRIMARY KEY,
    ch   TEXT NOT NULL UNIQUE
);

-- Bytes that are already printable ASCII/Latin-1 map to themselves...
INSERT INTO llm_byte_encoder(byte, ch)
SELECT b, chr(b)
  FROM generate_series(0, 255) AS g(b)
 WHERE b BETWEEN 33 AND 126
    OR b BETWEEN 161 AND 172
    OR b BETWEEN 174 AND 255;

-- ...and the remaining 68 (control characters, space, soft hyphen) are shifted
-- into the U+0100 block in ascending byte order, exactly as GPT-2 does.
INSERT INTO llm_byte_encoder(byte, ch)
SELECT b, chr(256 + (ROW_NUMBER() OVER (ORDER BY b))::INT - 1)
  FROM generate_series(0, 255) AS g(b)
 WHERE NOT (b BETWEEN 33 AND 126
         OR b BETWEEN 161 AND 172
         OR b BETWEEN 174 AND 255);

-- GPT-2's pre-tokenizer pattern, translated to POSIX ARE.  Splitting the input
-- into these chunks before applying BPE stops merges from running across word
-- boundaries.
CREATE OR REPLACE FUNCTION llm_pretokenize(text_in TEXT)
RETURNS TABLE(chunk TEXT, ord BIGINT) AS $$
    SELECT m.chunk[1], m.ord
      FROM regexp_matches(
               text_in,
               '''s|''t|''re|''ve|''m|''ll|''d'
               || '| ?[[:alpha:]]+'
               || '| ?[[:digit:]]+'
               || '| ?[^[:space:][:alpha:][:digit:]]+'
               || '|[[:space:]]+(?![^[:space:]])'
               || '|[[:space:]]+',
               'g') WITH ORDINALITY AS m(chunk, ord);
$$ LANGUAGE sql STABLE;

-- Byte-pair-encode one pre-tokenized chunk into token ids.
CREATE OR REPLACE FUNCTION llm_encode_chunk(chunk TEXT, p_model TEXT)
RETURNS INT[] AS $$
DECLARE
    symbols  TEXT[];
    merged   TEXT[];
    best_l   TEXT;
    best_r   TEXT;
    i        INT;
    n        INT;
BEGIN
    IF chunk IS NULL OR chunk = '' THEN
        RETURN ARRAY[]::INT[];
    END IF;

    -- One symbol per input byte, via the bytes_to_unicode mapping.
    SELECT array_agg(e.ch ORDER BY g.i)
      INTO symbols
      FROM generate_series(0, octet_length(convert_to(chunk, 'UTF8')) - 1) AS g(i)
      JOIN llm_byte_encoder e
        ON e.byte = get_byte(convert_to(chunk, 'UTF8'), g.i);

    -- Repeatedly merge the adjacent pair with the lowest merge rank.
    LOOP
        n := COALESCE(array_length(symbols, 1), 0);
        EXIT WHEN n < 2;

        best_l := NULL;
        best_r := NULL;

        SELECT symbols[p.i], symbols[p.i + 1]
          INTO best_l, best_r
          FROM generate_series(1, n - 1) AS p(i)
          JOIN llm_bpe_merges m
            ON m.model = p_model
           AND m."left" = symbols[p.i]
           AND m."right" = symbols[p.i + 1]
         ORDER BY m.rank
         LIMIT 1;

        EXIT WHEN best_l IS NULL;

        merged := ARRAY[]::TEXT[];
        i := 1;
        WHILE i <= n LOOP
            IF i < n AND symbols[i] = best_l AND symbols[i + 1] = best_r THEN
                merged := merged || (best_l || best_r);
                i := i + 2;
            ELSE
                merged := merged || symbols[i];
                i := i + 1;
            END IF;
        END LOOP;
        symbols := merged;
    END LOOP;

    RETURN COALESCE((
        SELECT array_agg(v.token_id ORDER BY s.ord)
          FROM unnest(symbols) WITH ORDINALITY AS s(sym, ord)
          JOIN llm_bpe_vocab v
            ON v.model = p_model
           AND v.token = s.sym
    ), ARRAY[]::INT[]);
END;
$$ LANGUAGE plpgsql STABLE;

CREATE OR REPLACE FUNCTION llm_encode(text_in TEXT, p_model TEXT)
RETURNS INT[] AS $$
DECLARE
    ids INT[] := ARRAY[]::INT[];
    rec RECORD;
BEGIN
    IF text_in IS NULL OR text_in = '' THEN
        RETURN ids;
    END IF;

    FOR rec IN
        SELECT t.chunk FROM llm_pretokenize(text_in) AS t ORDER BY t.ord
    LOOP
        ids := ids || llm_encode_chunk(rec.chunk, p_model);
    END LOOP;

    RETURN ids;
END;
$$ LANGUAGE plpgsql STABLE;

-- Inference-only counterpart to llm_materialize_params: llm_forward_gpt2 reads
-- the per-layer weights out of the llm_tensor cache, so the cache has to be
-- warm before a forward pass.  Unlike the training path this skips the autograd
-- runtime tables and the (large) embedding rows, which llm_embed and the final
-- projection read straight from llm_param_resolved.
CREATE OR REPLACE FUNCTION llm_materialize_inference_params(
    p_model TEXT,
    p_force BOOLEAN DEFAULT false)
RETURNS VOID AS $$
BEGIN
    IF NOT p_force
       AND EXISTS (SELECT 1 FROM llm_tensor_owner o WHERE o.model = p_model)
    THEN
        RETURN;
    END IF;

    DELETE FROM llm_tensor;

    INSERT INTO llm_tensor(name, data, requires_grad)
    SELECT p.name, p.data, false
      FROM llm_param_resolved p
     WHERE p.model = p_model
       AND p.name NOT IN ('wte', 'wpe')
    ON CONFLICT (name) DO UPDATE
        SET data = EXCLUDED.data,
            requires_grad = EXCLUDED.requires_grad;

    DELETE FROM llm_tensor_owner;
    INSERT INTO llm_tensor_owner(model) VALUES (p_model);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION llm_logits(
    token_ids INT[],
    model_name TEXT DEFAULT 'gpt2-small',
    n_layer INT DEFAULT NULL,
    n_head INT DEFAULT NULL,
    d_model INT DEFAULT NULL,
    vocab_size INT DEFAULT NULL,
    last_only BOOLEAN DEFAULT false)
RETURNS BYTEA AS $$
DECLARE
    seq_len INT := COALESCE(array_length(token_ids, 1), 0);
    x BYTEA;
    weight_matrix BYTEA;
    cfg llm_model_config%ROWTYPE;
    effective_n_layer INT := n_layer;
    effective_n_head INT := n_head;
    effective_d_model INT := d_model;
    effective_vocab INT := vocab_size;
BEGIN
    IF seq_len = 0 THEN
        RETURN ''::BYTEA;
    END IF;

    IF effective_n_layer IS NULL OR effective_n_head IS NULL
       OR effective_d_model IS NULL OR effective_vocab IS NULL THEN
        cfg := llm_get_model_config(model_name);
        effective_n_layer := COALESCE(effective_n_layer, cfg.n_layer);
        effective_n_head := COALESCE(effective_n_head, cfg.n_head);
        effective_d_model := COALESCE(effective_d_model, cfg.d_model);
        effective_vocab := COALESCE(effective_vocab, cfg.vocab);
    END IF;

    IF effective_d_model IS NULL THEN
        SELECT octet_length(p.data) / 4
          INTO effective_d_model
          FROM llm_param_resolved p
         WHERE p.model = model_name
           AND p.name = 'wte'
         ORDER BY p.token_id
         LIMIT 1;
    END IF;

    IF effective_d_model IS NULL THEN
        RAISE EXCEPTION 'Missing token embeddings for model %', model_name;
    END IF;

    IF effective_vocab IS NULL THEN
        SELECT COUNT(*)
          INTO effective_vocab
          FROM llm_param_resolved p
         WHERE p.model = model_name
           AND p.name = 'wte';
    END IF;

    PERFORM llm_materialize_inference_params(model_name);

    x := llm_embed(token_ids, model_name, effective_d_model);

    x := llm_forward_gpt2(
        x,
        model_name,
        effective_n_layer,
        effective_n_head,
        seq_len,
        effective_d_model,
        dropout_p => 0.0::float4,
        training => false);

    SELECT string_agg(p.data, ''::BYTEA ORDER BY p.token_id)
      INTO weight_matrix
      FROM llm_param_resolved p
     WHERE p.model = model_name
       AND p.name = 'wte';

    IF weight_matrix IS NULL THEN
        RAISE EXCEPTION 'Missing token embeddings for model %', model_name;
    END IF;

    -- Autoregressive sampling only ever needs the distribution for the token
    -- after the prompt, so projecting just the final hidden row turns the
    -- vocabulary matmul from seq_len x d_model x vocab into 1 x d_model x vocab.
    IF last_only THEN
        x := substring(x
                       FROM ((seq_len - 1) * effective_d_model * 4) + 1
                       FOR effective_d_model * 4);
        RETURN pg_llm_matmul(x, weight_matrix, 1, effective_d_model, effective_vocab);
    END IF;

    RETURN pg_llm_matmul(x, weight_matrix, seq_len, effective_d_model, effective_vocab);
END;
$$ LANGUAGE plpgsql;

-- Lossy UTF-8 decode. BPE tokens are byte sequences, so a sampled sequence is
-- not guaranteed to be well-formed text; invalid bytes become U+FFFD instead of
-- failing the whole decode.
CREATE OR REPLACE FUNCTION llm_bytes_to_text(buf BYTEA)
RETURNS TEXT AS $$
DECLARE
    n        INT := octet_length(buf);
    i        INT := 0;
    k        INT;
    lead     INT;
    cont     INT;
    width    INT;
    codepoint INT;
    valid    BOOLEAN;
    replacement TEXT := U&'\FFFD';
    parts    TEXT[] := ARRAY[]::TEXT[];
BEGIN
    WHILE i < n LOOP
        lead := get_byte(buf, i);

        IF lead < 128 THEN
            width := 1; codepoint := lead;
        ELSIF lead BETWEEN 194 AND 223 THEN
            width := 2; codepoint := lead - 192;
        ELSIF lead BETWEEN 224 AND 239 THEN
            width := 3; codepoint := lead - 224;
        ELSIF lead BETWEEN 240 AND 244 THEN
            width := 4; codepoint := lead - 240;
        ELSE
            width := 0; codepoint := 0;
        END IF;

        valid := width > 0 AND i + width <= n;

        IF valid THEN
            FOR k IN 1 .. width - 1 LOOP
                cont := get_byte(buf, i + k);
                IF cont < 128 OR cont > 191 THEN
                    valid := false;
                    EXIT;
                END IF;
                codepoint := codepoint * 64 + (cont - 128);
            END LOOP;
        END IF;

        -- Postgres cannot represent NUL in text, and surrogates are not scalar
        -- values, so both are treated as invalid here.
        IF valid AND (codepoint = 0
                      OR codepoint > 1114111
                      OR codepoint BETWEEN 55296 AND 57343) THEN
            valid := false;
        END IF;

        IF valid THEN
            parts := parts || chr(codepoint);
            i := i + width;
        ELSE
            parts := parts || replacement;
            i := i + 1;
        END IF;
    END LOOP;

    RETURN array_to_string(parts, '');
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT;

CREATE OR REPLACE FUNCTION llm_decode(ids INT[], p_model TEXT)
RETURNS TEXT AS $$
DECLARE
    symbols TEXT;
    buf     BYTEA;
    trimmed INT := 0;
BEGIN
    IF ids IS NULL OR array_length(ids, 1) IS NULL THEN
        RETURN '';
    END IF;

    SELECT string_agg(v.token, '' ORDER BY t.ord)
      INTO symbols
      FROM unnest(ids) WITH ORDINALITY AS t(id, ord)
      JOIN llm_bpe_vocab v
        ON v.model = p_model
       AND v.token_id = t.id;

    IF symbols IS NULL OR symbols = '' THEN
        RETURN '';
    END IF;

    -- Reverse the bytes_to_unicode mapping to recover the original byte stream.
    SELECT string_agg(set_byte('\x00'::BYTEA, 0, e.byte), ''::BYTEA ORDER BY c.ord)
      INTO buf
      FROM regexp_split_to_table(symbols, '') WITH ORDINALITY AS c(ch, ord)
      JOIN llm_byte_encoder e
        ON e.ch = c.ch;

    IF buf IS NULL THEN
        RETURN '';
    END IF;

    -- Fast path: the sequence is already well-formed text.
    BEGIN
        RETURN convert_from(buf, 'UTF8');
    EXCEPTION WHEN OTHERS THEN
        NULL;
    END;

    -- A partial generation commonly ends mid-codepoint. Dropping the incomplete
    -- tail keeps streaming output clean without introducing replacement chars.
    WHILE trimmed < 3 LOOP
        trimmed := trimmed + 1;
        IF octet_length(buf) <= trimmed THEN
            EXIT;
        END IF;
        BEGIN
            RETURN convert_from(substring(buf FROM 1 FOR octet_length(buf) - trimmed), 'UTF8');
        EXCEPTION WHEN OTHERS THEN
            NULL;
        END;
    END LOOP;

    -- Genuinely invalid bytes somewhere in the middle: decode what we can.
    RETURN llm_bytes_to_text(buf);
END;
$$ LANGUAGE plpgsql STABLE;

-- Single autoregressive step: score the sequence so far and sample the next
-- token id.  llm_generate/llm_generate_stream build on this, and it also lets a
-- client drive the loop itself when it wants each token as soon as it exists
-- (PL/pgSQL set-returning functions buffer their whole result before the first
-- row is visible, so llm_generate_stream cannot stream over the wire).
CREATE OR REPLACE FUNCTION llm_next_token(
    token_ids INT[],
    model_name TEXT DEFAULT 'gpt2-small',
    temperature FLOAT4 DEFAULT 1.0,
    topk INT DEFAULT 50,
    topp FLOAT4 DEFAULT 0.95)
RETURNS INT AS $$
BEGIN
    IF token_ids IS NULL OR array_length(token_ids, 1) IS NULL THEN
        RAISE EXCEPTION 'llm_next_token requires at least one token';
    END IF;

    RETURN pg_llm_sample(
               llm_logits(token_ids, model_name, last_only => true),
               temperature, topk, topp);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION llm_generate(
    prompt TEXT,
    max_tokens INT DEFAULT 64,
    temperature FLOAT4 DEFAULT 1.0,
    topk INT DEFAULT 50,
    topp FLOAT4 DEFAULT 0.95,
    model_name TEXT DEFAULT 'gpt2-small',
    eos_token INT DEFAULT 50256)
RETURNS TEXT AS $$
DECLARE
    ids INT[] := COALESCE(llm_encode(prompt, model_name), ARRAY[]::INT[]);
    next_id INT;
BEGIN
    FOR i IN 1..max_tokens LOOP
        next_id := llm_next_token(ids, model_name, temperature, topk, topp);
        ids := array_append(ids, next_id);
        EXIT WHEN next_id = eos_token;
    END LOOP;
    RETURN COALESCE(llm_decode(ids, model_name), '');
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION llm_generate_stream(
    prompt TEXT,
    max_tokens INT DEFAULT 64,
    temperature FLOAT4 DEFAULT 1.0,
    topk INT DEFAULT 50,
    topp FLOAT4 DEFAULT 0.95,
    model_name TEXT DEFAULT 'gpt2-small',
    eos_token INT DEFAULT 50256)
RETURNS TABLE(
    step INT,
    token_id INT,
    token TEXT,
    text TEXT,
    is_complete BOOLEAN)
AS $$
DECLARE
    ids INT[] := COALESCE(llm_encode(prompt, model_name), ARRAY[]::INT[]);
    next_id INT;
BEGIN
    step := 0;
    LOOP
        EXIT WHEN step >= max_tokens;
        step := step + 1;

        next_id := llm_next_token(ids, model_name, temperature, topk, topp);
        ids := array_append(ids, next_id);

        token_id := next_id;
        token := COALESCE(
            (
                SELECT v.token
                  FROM llm_bpe_vocab v
                 WHERE v.model = model_name
                   AND v.token_id = next_id
                 LIMIT 1
            ),
            ''
        );
        text := COALESCE(llm_decode(ids, model_name), '');
        is_complete := next_id = eos_token OR step >= max_tokens;
        RETURN NEXT;

        EXIT WHEN next_id = eos_token;
    END LOOP;

    RETURN;
END;
$$ LANGUAGE plpgsql;


CREATE FUNCTION pg_llm_sample(
    logits BYTEA,
    temperature FLOAT4 DEFAULT 1.0,
    topk INT DEFAULT 50,
    topp FLOAT4 DEFAULT 0.95)
RETURNS INT
AS 'MODULE_PATHNAME', 'pg_llm_sample'
LANGUAGE C STRICT;
