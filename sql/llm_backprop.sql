CREATE OR REPLACE FUNCTION llm_backprop(start_id INT)
RETURNS VOID AS $$
DECLARE
    node RECORD;
    grad BYTEA;
    a_id INT; b_id INT;
    a BYTEA; b BYTEA;
    m INT; k INT; n INT;
    dx BYTEA; dgamma BYTEA; dbeta BYTEA;
BEGIN
    -- seed gradient of final output = 1
    UPDATE llm_tensor_rt SET grad = pg_llm_ones_like(data) WHERE id=start_id;

    -- Replay the tape in reverse order.  Each op name recorded during the
    -- forward pass determines which gradient kernel we invoke.
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
            SELECT dx, dgamma, dbeta INTO dx, dgamma, dbeta
             FROM pg_llm_layernorm_backward(
                 (SELECT data FROM llm_tensor_rt WHERE id=node.inputs[1]),
                 (SELECT grad FROM llm_tensor_rt WHERE id=node.output),
                 (SELECT data FROM llm_tensor_rt WHERE id=(node.extra->>'gamma_id')::INT),
                 (node.extra->>'eps')::FLOAT4);

            UPDATE llm_tensor_rt
              SET grad = COALESCE(grad, pg_llm_zeros_like(data)) + dx
              WHERE id=node.inputs[1];
            UPDATE llm_tensor_rt
              SET grad = COALESCE(grad, pg_llm_zeros_like(data)) + dgamma
              WHERE id=(node.extra->>'gamma_id')::INT;
            UPDATE llm_tensor_rt
              SET grad = COALESCE(grad, pg_llm_zeros_like(data)) + dbeta
              WHERE id=(node.extra->>'beta_id')::INT;
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



