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
    target INT[]           -- shifted targets
);

CREATE OR REPLACE FUNCTION llm_loss(
    model TEXT,
    tokens INT[],
    targets INT[],
    n_layer INT,
    n_head INT,
    D INT,
    vocab INT)
RETURNS FLOAT4 AS $$
DECLARE
    x BYTEA;
    logits BYTEA;
    loss FLOAT4 := 0.0;
BEGIN
    -- 1. Embed tokens
    x := llm_embed(tokens, model, D);   -- we'll define this below

    -- 2. Forward pass through transformer
    x := llm_forward_gpt2(x, n_layer, n_head, array_length(tokens,1), D);

    -- 3. Final linear projection (tie weights with token_emb)
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
    beta1 FLOAT4,
    beta2 FLOAT4,
    eps FLOAT4,
    wd FLOAT4,
    lr_max FLOAT4,
    warmup INT,
    total_steps INT)
RETURNS FLOAT4 AS $$
DECLARE
    seq INT[];
    target INT[];
    step_count INT;
    lr FLOAT4;
    loss FLOAT4;
BEGIN
    SELECT tokens, target INTO seq, target FROM llm_dataset WHERE id=batch_id;
    SELECT MAX(step) + 1 INTO step_count FROM llm_param WHERE model=model;

    lr := pg_llm_lr_schedule(step_count, warmup, total_steps, lr_max);

    loss := llm_loss(model, seq, target, n_layer, n_head, D, vocab);

    -- In a full implementation, populate llm_param.grad here
    -- For now, assume gradients already in llm_param.grad.

    UPDATE llm_param
    SET (data, m, v, step) = (
        SELECT s.weight, s.m, s.v, step_count
        FROM pg_llm_adamw_step(
            data, grad, m, v,
            lr, beta1, beta2, eps, wd, step_count
        ) AS s
    )
    WHERE model = llm_param.model AND name = llm_param.name;

    INSERT INTO llm_train_log(model, step, lr, loss)
    VALUES(model, step_count, lr, loss);
    RETURN loss;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION llm_embed(tokens INT[], model TEXT, D INT)
RETURNS BYTEA AS $$
DECLARE
    out BYTEA;
BEGIN
    -- Flatten token embeddings
    SELECT string_agg(p.data::TEXT, '' ORDER BY t.ord)::BYTEA INTO out
    FROM unnest(tokens) WITH ORDINALITY AS t(token_id, ord)
    JOIN llm_param p
      ON p.model = model
     AND p.name = 'wte'
     AND p.token_id = t.token_id;

    IF out IS NULL THEN
        RAISE EXCEPTION 'Missing embeddings for one or more tokens in model %', model;
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
    beta1 FLOAT4 DEFAULT 0.9,
    beta2 FLOAT4 DEFAULT 0.999,
    eps FLOAT4 DEFAULT 1e-8,
    wd FLOAT4 DEFAULT 0.01,
    lr_max FLOAT4 DEFAULT 2.5e-4,
    warmup INT DEFAULT 2000)
RETURNS VOID AS $$
DECLARE
    loss FLOAT4;
BEGIN
    FOR i IN 1..n_steps LOOP
        loss := llm_train_step(
            model, (i % (SELECT COUNT(*) FROM llm_dataset))+1,
            n_layer, n_head, D, vocab,
            beta1, beta2, eps, wd,
            lr_max, warmup, n_steps);
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

-- store actual data buffers
CREATE UNLOGGED TABLE llm_tensor_rt (
    id SERIAL PRIMARY KEY,
    data BYTEA,
    grad BYTEA,            -- accumulated gradient
    shape INT[],
    requires_grad BOOL DEFAULT false
);

CREATE OR REPLACE FUNCTION llm_accumulate_grads(model TEXT)
RETURNS VOID AS $$
BEGIN
    UPDATE llm_param p
    SET grad = (
        SELECT grad FROM llm_tensor_rt t
        WHERE t.id = (SELECT id FROM llm_tensor_map WHERE model=p.model AND name=p.name)
    )
    WHERE model=p.model;
END;
$$ LANGUAGE plpgsql;

BEGIN;
  DELETE FROM llm_tape;
  DELETE FROM llm_tensor_rt;

  -- Forward pass with autograd enabled
  INSERT INTO llm_autograd_mode VALUES(true);
  PERFORM llm_loss('gpt2-small', seq, target, 12, 12, 768, 50257);

  -- Reverse pass
  PERFORM llm_backprop((SELECT MAX(id) FROM llm_tape));

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

PG_FUNCTION_INFO_V1(pg_llm_export_npz);
Datum pg_llm_export_npz(PG_FUNCTION_ARGS)
{
    text *path_t = PG_GETARG_TEXT_P(0);
    text *model_t= PG_GETARG_TEXT_P(1);
    char *path=text_to_cstring(path_t);
    char *model=text_to_cstring(model_t);

    SPI_connect();
    gzFile fp=gzopen(path,"wb");
    if(!fp) ereport(ERROR,(errmsg("cannot open %s",path)));

    SPI_execute("SELECT name,data FROM llm_param WHERE model=$1",true,0);

    for(uint64 i=0;i<SPI_processed;++i){
        HeapTuple t=SPI_tuptable->vals[i];
        char *name=TextDatumGetCString(SPI_getbinval(t,SPI_tuptable->tupdesc,1,NULL));
        bytea *b=(bytea*)DatumGetPointer(SPI_getbinval(t,SPI_tuptable->tupdesc,2,NULL));
        write_npz_entry(fp,name,(float*)VARDATA_ANY(b),VARHDRSZ, nbytes(b)/sizeof(float));
    }
    gzclose(fp);
    SPI_finish();
    PG_RETURN_VOID();
}
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
    topp FLOAT4 DEFAULT 0.95)
RETURNS TEXT AS $$
DECLARE
    ids INT[] := llm_encode(prompt,'gpt2-small');
    next_id INT;
BEGIN
    FOR i IN 1..max_tokens LOOP
        next_id := pg_llm_sample(llm_logits(ids), temperature, topk, topp);
        ids := array_append(ids,next_id);
        EXIT WHEN next_id=50256;
    END LOOP;
    RETURN llm_decode(ids,'gpt2-small');
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
