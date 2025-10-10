SET client_min_messages = warning;
TRUNCATE llm_param,
         llm_param_share,
         llm_tensor,
         llm_tensor_rt,
         llm_tensor_map,
         llm_tape;

INSERT INTO llm_param(model, name, token_id, data, grad, m, v, step)
VALUES
    ('model_a', 'shared', 0, decode('0000803f', 'hex'), NULL, NULL, NULL, 0),
    ('model_b', 'unique', 0, decode('00000000', 'hex'), NULL, NULL, NULL, 0);

SELECT llm_share_param('model_a', 'shared', 0, 'model_b', 'shared', 0);

SELECT count(*) AS model_b_shared_rows
FROM llm_param
WHERE model = 'model_b'
  AND name = 'shared';

SELECT encode(data, 'hex') AS resolved_shared
FROM llm_param_resolved
WHERE model = 'model_b'
  AND name = 'shared';

SELECT source_model, source_name, target_model, target_name
FROM llm_param_share
ORDER BY 1, 2, 3, 4;

SELECT llm_materialize_params('model_b');

UPDATE llm_tensor_rt
SET grad = decode('00000040', 'hex')
WHERE id IN (
    SELECT tensor_id
    FROM llm_tensor_map
    WHERE model = 'model_b'
      AND name = 'shared'
      AND token_id = 0
);

SELECT llm_accumulate_grads('model_b');

SELECT encode(grad, 'hex') AS shared_grad
FROM llm_param
WHERE model = 'model_a'
  AND name = 'shared';

SELECT llm_unshare_param('model_b', 'shared', 0);

SELECT encode(data, 'hex') AS unshared_data, step
FROM llm_param
WHERE model = 'model_b'
  AND name = 'shared';

SELECT count(*) AS remaining_shares
FROM llm_param_share
WHERE target_model = 'model_b';
