CREATE OR REPLACE FUNCTION llm_backprop(start_id INT, p_model TEXT)
RETURNS VOID AS $$
DECLARE
    node RECORD;
    grad BYTEA;
    m INT; k INT; n INT;
    b_grad BYTEA;
    grad_rows BYTEA;
    token_idx INT;
    bytes_per_row INT;
    wte_tensor_id INT;
    chunk BYTEA;
    has_mapping BOOLEAN;
    ln_dx BYTEA;
    ln_dgamma BYTEA;
    ln_dbeta BYTEA;
    attn_dx BYTEA;
    attn_dw_qkv BYTEA;
    attn_db_qkv BYTEA;
    attn_dw_o BYTEA;
    attn_db_o BYTEA;
    ln_gamma_id INT;
    ln_beta_id INT;
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

            SELECT EXISTS(
                       SELECT 1
                         FROM llm_tensor_map
                        WHERE tensor_id = node.inputs[2]
                          AND model = p_model) INTO has_mapping;

            IF NOT has_mapping THEN
                SELECT grad INTO b_grad FROM llm_tensor_rt WHERE id = node.inputs[2];

                IF b_grad IS NOT NULL THEN
                    grad_rows := pg_llm_transpose(b_grad, k, n);
                    bytes_per_row := k * 4;

                    FOR token_idx IN 0..(n-1) LOOP
                        SELECT tensor_id INTO wte_tensor_id
                          FROM llm_tensor_map
                         WHERE model = p_model
                           AND name = 'wte'
                           AND token_id = token_idx;

                        IF wte_tensor_id IS NULL THEN
                            CONTINUE;
                        END IF;

                        chunk := substring(grad_rows FROM token_idx * bytes_per_row + 1 FOR bytes_per_row);

                        UPDATE llm_tensor_rt
                           SET grad = CASE
                                          WHEN grad IS NULL THEN chunk
                                          ELSE pg_llm_add(grad, chunk)
                                      END
                         WHERE id = wte_tensor_id;
                    END LOOP;
                END IF;
            END IF;
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
            SELECT dx, dgamma, dbeta
              INTO ln_dx, ln_dgamma, ln_dbeta
              FROM pg_llm_layernorm_backward(
                  (SELECT data FROM llm_tensor_rt WHERE id=node.inputs[1]),
                  (SELECT grad FROM llm_tensor_rt WHERE id=node.output),
                  (SELECT data FROM llm_tensor_rt WHERE id=node.inputs[2]),
                  (node.extra->>'eps')::FLOAT4);

            UPDATE llm_tensor_rt
              SET grad = COALESCE(grad, pg_llm_zeros_like(data)) + ln_dx
              WHERE id = node.inputs[1];

            UPDATE llm_tensor_rt
              SET grad = COALESCE(grad, pg_llm_zeros_like(data)) + ln_dgamma
              WHERE id = node.inputs[2];

            UPDATE llm_tensor_rt
              SET grad = COALESCE(grad, pg_llm_zeros_like(data)) + ln_dbeta
              WHERE id = node.inputs[3];
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

            SELECT dx, dgamma, dbeta
              INTO ln_dx, ln_dgamma, ln_dbeta
              FROM pg_llm_layernorm_backward(
                  (SELECT data FROM llm_tensor_rt WHERE id=node.inputs[1]),
                  (SELECT grad FROM llm_tensor_rt WHERE id=node.output),
                  (SELECT data FROM llm_tensor_rt WHERE id=ln_gamma_id),
                  (node.extra->>'eps')::FLOAT4)
              AS t(dx BYTEA, dgamma BYTEA, dbeta BYTEA);

            UPDATE llm_tensor_rt
              SET grad = COALESCE(grad, pg_llm_zeros_like(data)) + ln_dx
              WHERE id = node.inputs[1];

            UPDATE llm_tensor_rt
              SET grad = COALESCE(grad, pg_llm_zeros_like(data)) + ln_dgamma
              WHERE id = ln_gamma_id;

            UPDATE llm_tensor_rt
              SET grad = COALESCE(grad, pg_llm_zeros_like(data)) + ln_dbeta
              WHERE id = ln_beta_id;
        ELSIF node.name='gelu' THEN
            UPDATE llm_tensor_rt
              SET grad = COALESCE(grad, pg_llm_zeros_like(data))
                        + pg_llm_gelu_backward((SELECT data FROM llm_tensor_rt WHERE id=node.inputs[1]),
                                               (SELECT grad FROM llm_tensor_rt WHERE id=node.output))
              WHERE id=node.inputs[1];
        ELSIF node.name='ones_like' THEN
            -- constant tensor; no gradient to propagate
            CONTINUE;
        ELSIF node.name='zeros_like' THEN
            -- constant tensor; no gradient to propagate
            CONTINUE;
        ELSIF node.name='transpose' THEN
            UPDATE llm_tensor_rt
              SET grad = COALESCE(grad, pg_llm_zeros_like(data))
                        + pg_llm_transpose(
                            (SELECT grad FROM llm_tensor_rt WHERE id=node.output),
                            (node.extra->>'cols')::INT,
                            (node.extra->>'rows')::INT)
              WHERE id = node.inputs[1];
        ELSIF node.name='attention' THEN
            SELECT dx, dw_qkv, db_qkv, dw_o, db_o
              INTO attn_dx, attn_dw_qkv, attn_db_qkv, attn_dw_o, attn_db_o
            FROM pg_llm_attention_backward(
                (SELECT data FROM llm_tensor_rt WHERE id=node.inputs[1]),
                (SELECT data FROM llm_tensor_rt WHERE id=node.inputs[2]),
                (SELECT data FROM llm_tensor_rt WHERE id=node.inputs[3]),
                (SELECT data FROM llm_tensor_rt WHERE id=node.inputs[4]),
                (SELECT data FROM llm_tensor_rt WHERE id=node.inputs[5]),
                (SELECT grad FROM llm_tensor_rt WHERE id=node.output),
                (node.extra->>'n_head')::INT,
                (node.extra->>'T')::INT,
                (node.extra->>'D')::INT);

            UPDATE llm_tensor_rt
              SET grad = COALESCE(grad, pg_llm_zeros_like(data)) + attn_dx
              WHERE id = node.inputs[1];

            UPDATE llm_tensor_rt
              SET grad = COALESCE(grad, pg_llm_zeros_like(data)) + attn_dw_qkv
              WHERE id = node.inputs[2];

            UPDATE llm_tensor_rt
              SET grad = COALESCE(grad, pg_llm_zeros_like(data)) + attn_db_qkv
              WHERE id = node.inputs[3];

            UPDATE llm_tensor_rt
              SET grad = COALESCE(grad, pg_llm_zeros_like(data)) + attn_dw_o
              WHERE id = node.inputs[4];

            UPDATE llm_tensor_rt
              SET grad = COALESCE(grad, pg_llm_zeros_like(data)) + attn_db_o
              WHERE id = node.inputs[5];
        -- Similar for layernorm, softmax, cross_entropy etc.
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;



