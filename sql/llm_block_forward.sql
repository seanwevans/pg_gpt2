CREATE OR REPLACE FUNCTION llm_block_forward(
    input BYTEA,
    w_qkv BYTEA,
    w_o BYTEA,
    w_fc BYTEA,
    w_proj BYTEA,
    ln1_g BYTEA,
    ln1_b BYTEA,
    ln2_g BYTEA,
    ln2_b BYTEA,
    n_head INT,
    T INT,
    D INT,
    eps FLOAT4 DEFAULT 1e-5)
RETURNS BYTEA AS $$
DECLARE
    x BYTEA := input;
    attn BYTEA;
    mlp BYTEA;
BEGIN
    -- 1. LayerNorm
    x := pg_llm_layernorm(x, ln1_g, ln1_b, eps);

    -- 2. Self-Attention
    attn := pg_llm_attention(x, w_qkv, w_o, n_head, T, D);
    x := pg_llm_add(input, attn);  -- residual 1

    -- 3. LayerNorm
    x := pg_llm_layernorm(x, ln2_g, ln2_b, eps);

    -- 4. Feed-Forward MLP
    mlp := pg_llm_matmul(x, w_fc, T, D, 4*D);
    mlp := pg_llm_gelu(mlp);
    mlp := pg_llm_matmul(mlp, w_proj, T, 4*D, D);

    x := pg_llm_add(x, mlp);       -- residual 2
    RETURN x;
END;
$$ LANGUAGE plpgsql STRICT;

CREATE OR REPLACE FUNCTION llm_forward_gpt2(input BYTEA, n_layer INT, n_head INT, T INT, D INT)
RETURNS BYTEA AS $$
DECLARE
    x BYTEA := input;
BEGIN
    FOR i IN 0..(n_layer-1) LOOP
        x := llm_block_forward(
            x,
            (SELECT data FROM llm_tensor WHERE name = format('h.%s.attn.c_attn.weight', i)),
            (SELECT data FROM llm_tensor WHERE name = format('h.%s.attn.c_proj.weight', i)),
            (SELECT data FROM llm_tensor WHERE name = format('h.%s.mlp.c_fc.weight', i)),
            (SELECT data FROM llm_tensor WHERE name = format('h.%s.mlp.c_proj.weight', i)),
            (SELECT data FROM llm_tensor WHERE name = format('h.%s.ln_1.weight', i)),
            (SELECT data FROM llm_tensor WHERE name = format('h.%s.ln_1.bias', i)),
            (SELECT data FROM llm_tensor WHERE name = format('h.%s.ln_2.weight', i)),
            (SELECT data FROM llm_tensor WHERE name = format('h.%s.ln_2.bias', i)),
            n_head, T, D);
    END LOOP;
    RETURN x;
END;
$$ LANGUAGE plpgsql;
