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
    attn := pg_llm_dropout(attn, dropout_p, training);
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
    mlp := pg_llm_dropout(mlp, dropout_p, training);

    x := pg_llm_add(residual2, mlp);       -- residual 2
    RETURN x;
END;
$$ LANGUAGE plpgsql STRICT;

CREATE OR REPLACE FUNCTION llm_forward_gpt2(
    input BYTEA,
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
    w_o BYTEA;
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
        SELECT data INTO w_o FROM llm_tensor WHERE name = format('h.%s.attn.c_proj.weight', i);
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

        x := llm_block_forward(
            x,
            w_qkv,
            w_o,
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
