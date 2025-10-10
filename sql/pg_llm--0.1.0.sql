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

-- Materialized tensors used during the forward pass (activations, cached weights)
CREATE UNLOGGED TABLE llm_tensor (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE,
    data BYTEA,
    shape INT[],
    requires_grad BOOL DEFAULT false
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
    tokens INT[],          -- 1024-token sequence
    target INT[],          -- shifted targets
    CHECK (array_length(tokens, 1) IS NULL OR array_length(tokens, 1) <= 1024),
    CHECK (array_length(target, 1) IS NULL OR array_length(target, 1) <= 1024),
    CHECK ((array_length(tokens, 1) IS NULL AND array_length(target, 1) IS NULL)
           OR array_length(tokens, 1) = array_length(target, 1))
);

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
        (SELECT data FROM llm_param p WHERE p.model = model AND p.name = 'ln_f.weight'),
        (SELECT data FROM llm_param p WHERE p.model = model AND p.name = 'ln_f.bias'),
        dropout_p => dropout_p,
        training => training);

    -- 3. Final linear projection (tie weights with token_emb).
    --    When this concatenated matrix is handed to the matmul kernel the
    --    runtime id must be registered via pg_llm_autograd_map_param so the
    --    logits gradient is accumulated back into each `wte` row.
    logits := pg_llm_matmul(x,
        (SELECT string_agg(p.data::TEXT, '' ORDER BY p.token_id)::BYTEA
         FROM llm_param p
         WHERE p.model = model AND p.name = 'wte'),
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
    beta1 FLOAT4,
    beta2 FLOAT4,
    eps FLOAT4,
    wd FLOAT4,
    lr_max FLOAT4,
    warmup INT,
    total_steps INT,
    grad_clip FLOAT4 DEFAULT NULL)
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
    FROM llm_param
    WHERE llm_param.model = model;

    lr := pg_llm_lr_schedule(step_count, warmup, total_steps, lr_max);
    -- Reset autograd state from the previous step
    DELETE FROM llm_tape;
    DELETE FROM llm_tensor_rt;
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
    PERFORM llm_accumulate_grads(model);

    UPDATE llm_param
    SET (data, m, v, grad, step) = (
        SELECT s.weight, s.m, s.v, NULL::BYTEA, step_count
        FROM pg_llm_adamw_step(
            data,
            CASE
                WHEN grad IS NULL OR grad_clip IS NULL OR grad_clip <= 0 THEN grad
                ELSE pg_llm_grad_clip(grad, grad_clip)
            END,
            m, v,
            lr, beta1, beta2, eps, wd, step_count
        ) AS s
    )
    WHERE llm_param.model = model;

    INSERT INTO llm_train_log(model, step, lr, loss)
    VALUES(model, step_count, lr, loss);

    -- Free runtime autograd state eagerly so the next step starts clean
    DELETE FROM llm_tape;
    DELETE FROM llm_tensor_rt;

    RETURN loss;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION llm_embed(tokens INT[], model TEXT, D INT)
RETURNS BYTEA AS $$
DECLARE
    out BYTEA;
    seq_len INT;
BEGIN
    seq_len := COALESCE(array_length(tokens, 1), 0);

    IF seq_len = 0 THEN
        RETURN ''::BYTEA;
    END IF;

    IF seq_len > 1024 THEN
        RAISE EXCEPTION 'Sequence length % exceeds GPT-2 maximum of 1024 positions', seq_len;
    END IF;

    -- Flatten summed token and positional embeddings
    SELECT string_agg(
               pg_llm_add(wte.data, wpe.data)::TEXT,
               '' ORDER BY t.ord
           )::BYTEA
      INTO out
      FROM unnest(tokens) WITH ORDINALITY AS t(token_id, ord)
      JOIN llm_param wte
        ON wte.model = model
       AND wte.name = 'wte'
       AND wte.token_id = t.token_id
      JOIN llm_param wpe
        ON wpe.model = model
       AND wpe.name = 'wpe'
       AND wpe.token_id = t.ord - 1;

    IF out IS NULL THEN
        RAISE EXCEPTION 'Missing token or positional embeddings for model %', model;
    END IF;

    RETURN out;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION llm_train(
    model TEXT,
    n_steps INT,
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
    grad_clip FLOAT4 DEFAULT NULL)
RETURNS VOID AS $$
DECLARE
    loss FLOAT4;
    dataset_ids INT[];
    dataset_size INT;
    idx INT := 1;
    batch_id INT;
BEGIN
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
            n_layer, n_head, D, vocab,
            dropout_p,
            beta1, beta2, eps, wd,
            lr_max, warmup, n_steps,
            grad_clip);

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

-- guard flag that toggles autograd recording
CREATE UNLOGGED TABLE llm_autograd_mode (
    flag BOOL NOT NULL
);

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

CREATE FUNCTION pg_llm_autograd_map_param(
    model TEXT,
    name TEXT,
    token_id INT,
    tensor BYTEA,
    dims INT[] DEFAULT NULL)
RETURNS VOID
AS 'MODULE_PATHNAME', 'pg_llm_autograd_map_param'
LANGUAGE C;

CREATE OR REPLACE FUNCTION llm_materialize_params(p_model TEXT)
RETURNS VOID AS $$
DECLARE
    rec RECORD;
    tensor_name TEXT;
BEGIN
    -- Clear cached tensors for this step
    DELETE FROM llm_tensor;
    DELETE FROM llm_tensor_map WHERE model = p_model;

    -- Copy parameters into the tensor cache and create runtime tensors
    FOR rec IN
        SELECT name, token_id, data
        FROM llm_param
        WHERE model = p_model
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

CREATE OR REPLACE FUNCTION llm_accumulate_grads(p_model TEXT)
RETURNS VOID AS $$
BEGIN
    -- Clear stale gradients for this model
    UPDATE llm_param
    SET grad = NULL
    WHERE model = p_model;

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
END;
$$ LANGUAGE plpgsql;

BEGIN;
  DELETE FROM llm_tape;
  DELETE FROM llm_tensor_rt;

  -- Forward pass with autograd enabled
  INSERT INTO llm_autograd_mode VALUES(true);
  PERFORM llm_loss('gpt2-small', seq, target, 12, 12, 768, 50257);

  -- Reverse pass
  PERFORM llm_backprop((SELECT MAX(id) FROM llm_tape), 'gpt2-small');

  -- Gradient accumulation
  PERFORM llm_accumulate_grads('gpt2-small');

  -- Optimizer update
  PERFORM llm_train_step(...);

COMMIT;

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
                        (SELECT MAX(step) FROM llm_param WHERE model=model));
    n BIGINT;
BEGIN
    PERFORM pg_llm_export_npz(path, model);
    SELECT COUNT(*) INTO n FROM llm_param WHERE model=model;
    INSERT INTO llm_checkpoint(model,step,n_params,file_path,note)
    VALUES(model,(SELECT MAX(step) FROM llm_param WHERE model=model),
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
    model TEXT,
    token_id INT PRIMARY KEY,
    token TEXT,               -- raw text form
    score FLOAT4,             -- optional rank / freq
    bytes BYTEA               -- UTF-8 representation
);

CREATE TABLE llm_bpe_merges (
    model TEXT,
    rank INT PRIMARY KEY,
    left TEXT,
    right TEXT,
    pair TEXT                 -- "left right"
);

CREATE FUNCTION pg_llm_load_bpe_vocab(path TEXT, model TEXT)
RETURNS void
AS 'MODULE_PATHNAME', 'pg_llm_load_bpe_vocab'
LANGUAGE C STRICT;

CREATE FUNCTION pg_llm_load_bpe_merges(path TEXT, model TEXT)
RETURNS void
AS 'MODULE_PATHNAME', 'pg_llm_load_bpe_merges'
LANGUAGE C STRICT;

SELECT pg_llm_load_bpe_vocab('/mnt/gpt2/vocab.json','gpt2-small');
SELECT pg_llm_load_bpe_merges('/mnt/gpt2/merges.txt','gpt2-small');

CREATE OR REPLACE FUNCTION llm_encode(text_in TEXT, model TEXT)
RETURNS INT[] AS $$
DECLARE
    tokens TEXT[];
    pairs TEXT[];
    merged TEXT;
    done BOOL := false;
BEGIN
    -- split into UTF-8 bytes
    SELECT array_agg(chr(get_byte(t::bytea,i)))
    INTO tokens
    FROM generate_series(0,length(t::bytea)-1) i, (SELECT convert_to(text_in,'UTF8') t) _;
    
    WHILE NOT done LOOP
        pairs := ARRAY(
            SELECT format('%s %s',tokens[i],tokens[i+1])
            FROM generate_series(1,array_length(tokens,1)-1) g(i)
        );
        SELECT pair INTO merged
        FROM llm_bpe_merges WHERE model=model AND pair=ANY(pairs)
        ORDER BY rank LIMIT 1;
        IF merged IS NULL THEN
            done := true;
        ELSE
            tokens := (
                SELECT array_agg(CASE
                    WHEN i<array_length(tokens,1)
                      AND format('%s %s',tokens[i],tokens[i+1])=merged
                    THEN split_part(merged,' ',1)||split_part(merged,' ',2)
                    ELSE tokens[i]
                END ORDER BY i)
                FROM generate_series(1,array_length(tokens,1)) g(i)
            );
        END IF;
    END LOOP;

    RETURN ARRAY(
        SELECT token_id FROM llm_bpe_vocab
        WHERE token=ANY(tokens) ORDER BY array_position(tokens,token)
    );
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION llm_logits(
    token_ids INT[],
    model_name TEXT DEFAULT 'gpt2-small',
    n_layer INT DEFAULT 12,
    n_head INT DEFAULT 12,
    d_model INT DEFAULT NULL,
    vocab_size INT DEFAULT NULL)
RETURNS BYTEA AS $$
DECLARE
    seq_len INT := COALESCE(array_length(token_ids, 1), 0);
    x BYTEA;
    weight_matrix BYTEA;
BEGIN
    IF seq_len = 0 THEN
        RETURN ''::BYTEA;
    END IF;

    IF d_model IS NULL THEN
        SELECT octet_length(p.data) / 4
          INTO d_model
          FROM llm_param p
         WHERE p.model = model_name
           AND p.name = 'wte'
         ORDER BY p.token_id
         LIMIT 1;
    END IF;

    IF d_model IS NULL THEN
        RAISE EXCEPTION 'Missing token embeddings for model %', model_name;
    END IF;

    IF vocab_size IS NULL THEN
        SELECT COUNT(*)
          INTO vocab_size
          FROM llm_param p
         WHERE p.model = model_name
           AND p.name = 'wte';
    END IF;

    x := llm_embed(token_ids, model_name, d_model);

    x := llm_forward_gpt2(
        x,
        model_name,
        n_layer,
        n_head,
        seq_len,
        d_model,
        dropout_p => 0.0::float4,
        training => false);

    SELECT string_agg(p.data::TEXT, '' ORDER BY p.token_id)::BYTEA
      INTO weight_matrix
      FROM llm_param p
     WHERE p.model = model_name
       AND p.name = 'wte';

    IF weight_matrix IS NULL THEN
        RAISE EXCEPTION 'Missing token embeddings for model %', model_name;
    END IF;

    RETURN pg_llm_matmul(x, weight_matrix, seq_len, d_model, vocab_size);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION llm_decode(ids INT[], model TEXT)
RETURNS TEXT AS $$
DECLARE
    s TEXT := '';
BEGIN
    SELECT string_agg(token,'') INTO s
    FROM llm_bpe_vocab WHERE model=model AND token_id=ANY(ids)
    ORDER BY array_position(ids,token_id);
    RETURN s;
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
        next_id := pg_llm_sample(llm_logits(ids, model_name), temperature, topk, topp);
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

        next_id := pg_llm_sample(llm_logits(ids, model_name), temperature, topk, topp);
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
