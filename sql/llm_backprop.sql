CREATE OR REPLACE FUNCTION llm_backprop(start_id INT)
RETURNS VOID AS $$
DECLARE
    node RECORD;
    grad BYTEA;
    a_id INT; b_id INT;
    a BYTEA; b BYTEA;
    m INT; k INT; n INT;
    dx BYTEA;
    dw_qkv BYTEA;
    db_qkv BYTEA;
    dw_o BYTEA;
    db_o BYTEA;
BEGIN
    -- seed gradient of final output = 1
    UPDATE llm_tensor_rt SET grad = pg_llm_ones_like(data) WHERE id=start_id;

    FOR node IN SELECT * FROM llm_tape ORDER BY id DESC LOOP
        IF node.name='add' THEN
            UPDATE llm_tensor_rt
              SET grad = COALESCE(grad, pg_llm_zeros_like(data))
                        + (SELECT grad FROM llm_tensor_rt WHERE id=node.output)
              WHERE id = ANY(node.inputs);
        ELSIF node.name='matmul' THEN
            SELECT data INTO a FROM llm_tensor_rt WHERE id=node.inputs[1];
            SELECT data INTO b FROM llm_tensor_rt WHERE id=node.inputs[2];
            SELECT grad INTO grad FROM llm_tensor_rt WHERE id=node.output;

            SELECT (node.extra->>'m')::INT, (node.extra->>'k')::INT, (node.extra->>'n')::INT INTO m,k,n;

            -- dA = dY @ Bᵀ, dB = Aᵀ @ dY
            UPDATE llm_tensor_rt
              SET grad = COALESCE(grad, pg_llm_zeros_like(data))
                        + pg_llm_matmul(grad, pg_llm_transpose(b, k, n), m,n,k)
              WHERE id=node.inputs[1];
            UPDATE llm_tensor_rt
              SET grad = COALESCE(grad, pg_llm_zeros_like(data))
                        + pg_llm_matmul(pg_llm_transpose(a, m, k), grad, k,m,n)
              WHERE id=node.inputs[2];
        ELSIF node.name='softmax' THEN
            UPDATE llm_tensor_rt
              SET grad = pg_llm_softmax_backward(
                  (SELECT data FROM llm_tensor_rt WHERE id=node.output),
                  (SELECT grad FROM llm_tensor_rt WHERE id=node.output))
              WHERE id = node.inputs[1];

        ELSIF node.name='cross_entropy' THEN
            UPDATE llm_tensor_rt
              SET grad = COALESCE(grad, pg_llm_zeros_like(data))
                        + pg_llm_cross_entropy_backward(
                            (SELECT data FROM llm_tensor_rt WHERE id=node.inputs[1]),
                            (node.extra->>'target')::INT)
              WHERE id = node.inputs[1];

        ELSIF node.name='dropout' THEN
            UPDATE llm_tensor_rt
              SET grad = COALESCE(grad, pg_llm_zeros_like(data))
                        + pg_llm_dropout_backward(
                            (SELECT data FROM llm_tensor_rt WHERE id=node.inputs[1]),
                            (SELECT data FROM llm_tensor_rt WHERE id=node.output),
                            (SELECT grad FROM llm_tensor_rt WHERE id=node.output),
                            COALESCE((node.extra->>'p')::FLOAT4, 0.0::FLOAT4),
                            COALESCE((node.extra->>'training')::BOOLEAN, true))
              WHERE id=node.inputs[1];

        ELSIF node.name='layernorm' THEN
            PERFORM (SELECT dx, dgamma, dbeta
             FROM pg_llm_layernorm_backward(
                 (SELECT data FROM llm_tensor_rt WHERE id=node.inputs[1]),
                 (SELECT grad FROM llm_tensor_rt WHERE id=node.output),
                 (SELECT data FROM llm_tensor_rt WHERE id=node.extra->>'gamma_id'),
                 (node.extra->>'eps')::FLOAT4));
    -- accumulate dx→input.grad, dγ→γ.grad, dβ→β.grad
        ELSIF node.name='attention' THEN
            SELECT dx, dw_qkv, db_qkv, dw_o, db_o
              INTO dx, dw_qkv, db_qkv, dw_o, db_o
              FROM pg_llm_attention_backward(
                  (SELECT data FROM llm_tensor_rt WHERE id=node.inputs[1]),
                  (SELECT data FROM llm_tensor_rt WHERE id=node.inputs[2]),
                  (SELECT data FROM llm_tensor_rt WHERE id=node.inputs[3]),
                  (SELECT data FROM llm_tensor_rt WHERE id=node.inputs[4]),
                  (SELECT data FROM llm_tensor_rt WHERE id=node.inputs[5]),
                  (SELECT grad FROM llm_tensor_rt WHERE id=node.output),
                  (node.extra->>'n_head')::INT,
                  (node.extra->>'T')::INT,
                  (node.extra->>'D')::INT)
              AS t(dx BYTEA, dw_qkv BYTEA, db_qkv BYTEA, dw_o BYTEA, db_o BYTEA);

            UPDATE llm_tensor_rt
              SET grad = COALESCE(grad, pg_llm_zeros_like(data)) + dx
              WHERE id = node.inputs[1];
            UPDATE llm_tensor_rt
              SET grad = COALESCE(grad, pg_llm_zeros_like(data)) + dw_qkv
              WHERE id = node.inputs[2];
            UPDATE llm_tensor_rt
              SET grad = COALESCE(grad, pg_llm_zeros_like(data)) + db_qkv
              WHERE id = node.inputs[3];
            UPDATE llm_tensor_rt
              SET grad = COALESCE(grad, pg_llm_zeros_like(data)) + dw_o
              WHERE id = node.inputs[4];
            UPDATE llm_tensor_rt
              SET grad = COALESCE(grad, pg_llm_zeros_like(data)) + db_o
              WHERE id = node.inputs[5];
        ELSIF node.name='gelu' THEN
            UPDATE llm_tensor_rt
              SET grad = COALESCE(grad, pg_llm_zeros_like(data))
                        + pg_llm_gelu_backward((SELECT data FROM llm_tensor_rt WHERE id=node.inputs[1]),
                                               (SELECT grad FROM llm_tensor_rt WHERE id=node.output))
              WHERE id=node.inputs[1];
        -- Similar for layernorm, softmax, cross_entropy etc.
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;



