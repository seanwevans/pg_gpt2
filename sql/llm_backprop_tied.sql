SET extra_float_digits = 3;
\ir :abs_srcdir/llm_backprop.sql

DROP FUNCTION IF EXISTS arr_to_bytea(float4[]);
CREATE FUNCTION arr_to_bytea(vals float4[])
RETURNS bytea
LANGUAGE SQL
AS $$
    SELECT string_agg(le_bytes, ''::bytea ORDER BY ord)
    FROM unnest(vals) WITH ORDINALITY AS t(val, ord)
    CROSS JOIN LATERAL (
        SELECT set_byte(
                   set_byte(
                       set_byte(
                           set_byte('\x00000000'::bytea, 0, get_byte(be, 3)),
                           1, get_byte(be, 2)),
                       2, get_byte(be, 1)),
                   3, get_byte(be, 0)) AS le_bytes
        FROM (SELECT pg_catalog.float4send(val) AS be) s
    );
$$;

DROP FUNCTION IF EXISTS pg_llm_ones_like(BYTEA);
CREATE FUNCTION pg_llm_ones_like(input BYTEA)
RETURNS BYTEA
AS '/workspace/pg_gpt2/pg_llm', 'pg_llm_ones_like'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS pg_llm_zeros_like(BYTEA);
CREATE FUNCTION pg_llm_zeros_like(input BYTEA)
RETURNS BYTEA
AS '/workspace/pg_gpt2/pg_llm', 'pg_llm_zeros_like'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS pg_llm_add(BYTEA, BYTEA);
CREATE FUNCTION pg_llm_add(a BYTEA, b BYTEA)
RETURNS BYTEA
AS '/workspace/pg_gpt2/pg_llm', 'pg_llm_add'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS pg_llm_transpose(BYTEA, INT, INT);
CREATE FUNCTION pg_llm_transpose(input BYTEA, rows INT, cols INT)
RETURNS BYTEA
AS '/workspace/pg_gpt2/pg_llm', 'pg_llm_transpose'
LANGUAGE C STRICT;

DROP FUNCTION IF EXISTS pg_llm_matmul(BYTEA, BYTEA, INT, INT, INT);
CREATE FUNCTION pg_llm_matmul(a BYTEA, b BYTEA, m INT, k INT, n INT)
RETURNS BYTEA
AS '/workspace/pg_gpt2/pg_llm', 'pg_llm_matmul'
LANGUAGE C STRICT;

DROP TABLE IF EXISTS llm_tape;
CREATE UNLOGGED TABLE llm_tape (
    id SERIAL PRIMARY KEY,
    name TEXT,
    inputs INT[],
    output INT,
    extra JSONB
);

DROP TABLE IF EXISTS llm_tensor_rt;
CREATE UNLOGGED TABLE llm_tensor_rt (
    id SERIAL PRIMARY KEY,
    data BYTEA,
    grad BYTEA,
    shape INT[],
    requires_grad BOOL DEFAULT false
);

DROP TABLE IF EXISTS llm_tensor_map;
CREATE UNLOGGED TABLE llm_tensor_map (
    model TEXT,
    name TEXT,
    token_id INT NOT NULL,
    tensor_id INT,
    PRIMARY KEY (model, name, token_id)
);

TRUNCATE llm_tape;
TRUNCATE llm_tensor_rt;
TRUNCATE llm_tensor_map;

CREATE TEMP TABLE wte_values(token_id INT, bytes BYTEA);
INSERT INTO wte_values(token_id, bytes) VALUES
    (0, arr_to_bytea(ARRAY[0.1::float4, 0.2::float4])),
    (1, arr_to_bytea(ARRAY[0.3::float4, 0.4::float4])),
    (2, arr_to_bytea(ARRAY[0.5::float4, 0.6::float4]));

INSERT INTO llm_tensor_rt(id, data, grad, shape, requires_grad)
SELECT token_id + 1, bytes, NULL, NULL, true
FROM wte_values
ORDER BY token_id;

INSERT INTO llm_tensor_map(model, name, token_id, tensor_id)
SELECT 'gpt2-small', 'wte', token_id, token_id + 1
FROM wte_values
ORDER BY token_id;

INSERT INTO llm_tensor_rt(id, data, grad, shape, requires_grad)
SELECT 5, string_agg(bytes::text, '' ORDER BY token_id)::bytea, NULL, NULL, true
FROM wte_values;

INSERT INTO llm_tensor_rt(id, data, grad, shape, requires_grad)
VALUES (4, arr_to_bytea(ARRAY[1::float4, 2::float4, 3::float4, 4::float4]), NULL, NULL, true);

INSERT INTO llm_tensor_rt(id, data, grad, shape, requires_grad)
VALUES (7, arr_to_bytea(ARRAY[1::float4, 2::float4, 3::float4, 4::float4, 5::float4, 6::float4]), NULL, NULL, true);

INSERT INTO llm_tensor_rt(id, data, grad, shape, requires_grad)
SELECT 6, pg_llm_matmul(
           (SELECT data FROM llm_tensor_rt WHERE id = 4),
           (SELECT data FROM llm_tensor_rt WHERE id = 5),
           2, 2, 3),
       NULL, NULL, true;

INSERT INTO llm_tensor_rt(id, data, grad, shape, requires_grad)
SELECT 8, pg_llm_matmul(
           (SELECT data FROM llm_tensor_rt WHERE id = 6),
           (SELECT data FROM llm_tensor_rt WHERE id = 7),
           2, 3, 2),
       NULL, NULL, true;

INSERT INTO llm_tape(id, name, inputs, output, extra)
VALUES
    (1, 'matmul', ARRAY[4, 5], 6, jsonb_build_object('m', 2, 'k', 2, 'n', 3)),
    (2, 'matmul', ARRAY[6, 7], 8, jsonb_build_object('m', 2, 'k', 3, 'n', 2));

SELECT llm_backprop(8, 'gpt2-small');

WITH expected AS (
    SELECT * FROM (VALUES
        (0, arr_to_bytea(ARRAY[12::float4, 18::float4])),
        (1, arr_to_bytea(ARRAY[28::float4, 42::float4])),
        (2, arr_to_bytea(ARRAY[44::float4, 66::float4]))
    ) AS t(token_id, bytes)
)
SELECT format('token %s', expected.token_id) AS token_label,
       encode(rt.grad, 'hex') AS grad_hex,
       encode(expected.bytes, 'hex') AS expected_hex,
       rt.grad = expected.bytes AS matches
FROM expected
JOIN llm_tensor_map map
  ON map.model = 'gpt2-small'
 AND map.name = 'wte'
 AND map.token_id = expected.token_id
JOIN llm_tensor_rt rt
  ON rt.id = map.tensor_id
ORDER BY expected.token_id;
