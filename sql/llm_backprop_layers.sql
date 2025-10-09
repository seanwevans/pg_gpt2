SET extra_float_digits = 3;
\ir :abs_srcdir/llm_backprop.sql

CREATE OR REPLACE FUNCTION arr_to_bytea(vals float4[])
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

CREATE OR REPLACE FUNCTION pg_llm_add(a BYTEA, b BYTEA)
RETURNS BYTEA
AS '/workspace/pg_gpt2/pg_llm', 'pg_llm_add'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION pg_llm_gelu(a BYTEA)
RETURNS BYTEA
AS '/workspace/pg_gpt2/pg_llm', 'pg_llm_gelu'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION pg_llm_gelu_backward(x BYTEA, dy BYTEA)
RETURNS BYTEA
AS '/workspace/pg_gpt2/pg_llm', 'pg_llm_gelu_backward'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION pg_llm_softmax(a BYTEA)
RETURNS BYTEA
AS '/workspace/pg_gpt2/pg_llm', 'pg_llm_softmax'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION pg_llm_softmax_backward(y BYTEA, dy BYTEA)
RETURNS BYTEA
AS '/workspace/pg_gpt2/pg_llm', 'pg_llm_softmax_backward'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION pg_llm_cross_entropy(logits BYTEA, target INT)
RETURNS FLOAT4
AS '/workspace/pg_gpt2/pg_llm', 'pg_llm_cross_entropy'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION pg_llm_cross_entropy_backward(logits BYTEA, target INT)
RETURNS BYTEA
AS '/workspace/pg_gpt2/pg_llm', 'pg_llm_cross_entropy_backward'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION pg_llm_dropout(input BYTEA, p FLOAT4, training BOOLEAN)
RETURNS BYTEA
AS '/workspace/pg_gpt2/pg_llm', 'pg_llm_dropout'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION pg_llm_dropout_backward(input BYTEA, output BYTEA, grad BYTEA, p FLOAT4, training BOOLEAN)
RETURNS BYTEA
AS '/workspace/pg_gpt2/pg_llm', 'pg_llm_dropout_backward'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION pg_llm_ones_like(src BYTEA)
RETURNS BYTEA
AS '/workspace/pg_gpt2/pg_llm', 'pg_llm_ones_like'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION pg_llm_transpose(src BYTEA, rows INT, cols INT)
RETURNS BYTEA
AS '/workspace/pg_gpt2/pg_llm', 'pg_llm_transpose'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION pg_llm_matmul(a BYTEA, b BYTEA, m INT, k INT, n INT)
RETURNS BYTEA
AS '/workspace/pg_gpt2/pg_llm', 'pg_llm_matmul'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION pg_llm_layernorm(x BYTEA, gamma BYTEA, beta BYTEA, eps FLOAT4)
RETURNS BYTEA
AS '/workspace/pg_gpt2/pg_llm', 'pg_llm_layernorm'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION pg_llm_layernorm_backward(x BYTEA, dy BYTEA, gamma BYTEA, eps FLOAT4)
RETURNS TABLE (dx BYTEA, dgamma BYTEA, dbeta BYTEA)
AS '/workspace/pg_gpt2/pg_llm', 'pg_llm_layernorm_backward'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION pg_llm_attention(x BYTEA, w_qkv BYTEA, b_qkv BYTEA, w_o BYTEA, b_o BYTEA, n_head INT, T INT, D INT)
RETURNS BYTEA
AS '/workspace/pg_gpt2/pg_llm', 'pg_llm_attention'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION pg_llm_attention_backward(x BYTEA, w_qkv BYTEA, b_qkv BYTEA, w_o BYTEA, b_o BYTEA, grad BYTEA, n_head INT, T INT, D INT)
RETURNS TABLE (dx BYTEA, dw_qkv BYTEA, db_qkv BYTEA, dw_o BYTEA, db_o BYTEA)
AS '/workspace/pg_gpt2/pg_llm', 'pg_llm_attention_backward'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION pg_llm_mlp_backward(x BYTEA, w_fc BYTEA, b_fc BYTEA, w_proj BYTEA, b_proj BYTEA, grad BYTEA, T INT, D INT)
RETURNS TABLE (dx BYTEA, dw_fc BYTEA, db_fc BYTEA, dw_proj BYTEA, db_proj BYTEA)
AS '/workspace/pg_gpt2/pg_llm', 'pg_llm_mlp_backward'
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

DROP TABLE IF EXISTS llm_autograd_mode;
CREATE UNLOGGED TABLE llm_autograd_mode (flag BOOL NOT NULL);

DELETE FROM llm_autograd_mode;
INSERT INTO llm_autograd_mode(flag) VALUES(true);

\echo 'add backward'
TRUNCATE llm_tape;
TRUNCATE llm_tensor_rt;
TRUNCATE llm_tensor_map;
WITH fixture AS (
    SELECT arr_to_bytea(ARRAY[1::float4, 2::float4, 3::float4, 4::float4]) AS a,
           arr_to_bytea(ARRAY[5::float4, 6::float4, 7::float4, 8::float4]) AS b
),
forward AS (
    SELECT pg_llm_add(a, b) AS y FROM fixture
),
backprop AS (
    SELECT llm_backprop((SELECT output FROM llm_tape ORDER BY id DESC LIMIT 1), 'test-model')
),
node AS (
    SELECT * FROM llm_tape ORDER BY id DESC LIMIT 1
)
SELECT format('input_%s', idx) AS label,
       (SELECT grad FROM llm_tensor_rt WHERE id = inp.tensor_id) =
       pg_llm_ones_like((SELECT data FROM llm_tensor_rt WHERE id = inp.tensor_id)) AS matches
FROM node,
     LATERAL unnest(node.inputs) WITH ORDINALITY AS inp(tensor_id, idx)
ORDER BY idx;

\echo 'matmul backward'
TRUNCATE llm_tape;
TRUNCATE llm_tensor_rt;
TRUNCATE llm_tensor_map;
WITH fixture AS (
    SELECT arr_to_bytea(ARRAY[1::float4, 2::float4, 3::float4, 4::float4]) AS a,
           arr_to_bytea(ARRAY[0.5::float4, -1::float4, 1.5::float4, 2::float4]) AS b
),
forward AS (
    SELECT pg_llm_matmul(a, b, 2, 2, 2) AS y FROM fixture
),
backprop AS (
    SELECT llm_backprop((SELECT output FROM llm_tape ORDER BY id DESC LIMIT 1), 'test-model')
),
node AS (
    SELECT * FROM llm_tape ORDER BY id DESC LIMIT 1
)
SELECT 'dA matches' AS label,
       (SELECT grad FROM llm_tensor_rt WHERE id = node.inputs[1]) =
       pg_llm_matmul(
           (SELECT grad FROM llm_tensor_rt WHERE id = node.output),
           pg_llm_transpose((SELECT data FROM llm_tensor_rt WHERE id = node.inputs[2]),
                            (node.extra->>'k')::INT,
                            (node.extra->>'n')::INT),
           (node.extra->>'m')::INT,
           (node.extra->>'n')::INT,
           (node.extra->>'k')::INT) AS matches
FROM node
UNION ALL
SELECT 'dB matches' AS label,
       (SELECT grad FROM llm_tensor_rt WHERE id = node.inputs[2]) =
       pg_llm_matmul(
           pg_llm_transpose((SELECT data FROM llm_tensor_rt WHERE id = node.inputs[1]),
                            (node.extra->>'m')::INT,
                            (node.extra->>'k')::INT),
           (SELECT grad FROM llm_tensor_rt WHERE id = node.output),
           (node.extra->>'k')::INT,
           (node.extra->>'m')::INT,
           (node.extra->>'n')::INT) AS matches
FROM node;

\echo 'softmax backward'
TRUNCATE llm_tape;
TRUNCATE llm_tensor_rt;
TRUNCATE llm_tensor_map;
WITH fixture AS (
    SELECT arr_to_bytea(ARRAY[0.1::float4, 0.2::float4, -0.3::float4, 0.5::float4]) AS logits
),
forward AS (
    SELECT pg_llm_softmax(logits) AS probs FROM fixture
),
backprop AS (
    SELECT llm_backprop((SELECT output FROM llm_tape ORDER BY id DESC LIMIT 1), 'test-model')
),
node AS (
    SELECT * FROM llm_tape ORDER BY id DESC LIMIT 1
)
SELECT 'softmax grad matches' AS label,
       (SELECT grad FROM llm_tensor_rt WHERE id = node.inputs[1]) =
       pg_llm_softmax_backward(
           (SELECT data FROM llm_tensor_rt WHERE id = node.output),
           (SELECT grad FROM llm_tensor_rt WHERE id = node.output)) AS matches
FROM node;

\echo 'cross entropy backward'
TRUNCATE llm_tape;
TRUNCATE llm_tensor_rt;
TRUNCATE llm_tensor_map;
WITH fixture AS (
    SELECT arr_to_bytea(ARRAY[0.25::float4, -0.5::float4, 0.75::float4]) AS logits,
           3 AS target
),
forward AS (
    SELECT pg_llm_cross_entropy(logits, target - 1) AS loss FROM fixture
),
backprop AS (
    SELECT llm_backprop((SELECT output FROM llm_tape ORDER BY id DESC LIMIT 1), 'test-model')
),
node AS (
    SELECT * FROM llm_tape ORDER BY id DESC LIMIT 1
)
SELECT 'cross entropy grad matches' AS label,
       (SELECT grad FROM llm_tensor_rt WHERE id = node.inputs[1]) =
       pg_llm_cross_entropy_backward(
           (SELECT data FROM llm_tensor_rt WHERE id = node.inputs[1]),
           (node.extra->>'target')::INT) AS matches
FROM node;

\echo 'dropout backward'
TRUNCATE llm_tape;
TRUNCATE llm_tensor_rt;
TRUNCATE llm_tensor_map;
WITH fixture AS (
    SELECT arr_to_bytea(ARRAY[0.2::float4, -0.4::float4, 0.6::float4, -0.8::float4]) AS x
),
forward AS (
    SELECT pg_llm_dropout(x, 0.0, true) AS y FROM fixture
),
backprop AS (
    SELECT llm_backprop((SELECT output FROM llm_tape ORDER BY id DESC LIMIT 1), 'test-model')
),
node AS (
    SELECT * FROM llm_tape ORDER BY id DESC LIMIT 1
)
SELECT 'dropout grad matches' AS label,
       (SELECT grad FROM llm_tensor_rt WHERE id = node.inputs[1]) =
       pg_llm_dropout_backward(
           (SELECT data FROM llm_tensor_rt WHERE id = node.inputs[1]),
           (SELECT data FROM llm_tensor_rt WHERE id = node.output),
           (SELECT grad FROM llm_tensor_rt WHERE id = node.output),
           COALESCE((node.extra->>'p')::FLOAT4, 0.0::FLOAT4),
           COALESCE((node.extra->>'training')::BOOLEAN, true)) AS matches
FROM node;

\echo 'layernorm backward'
TRUNCATE llm_tape;
TRUNCATE llm_tensor_rt;
TRUNCATE llm_tensor_map;
WITH fixture AS (
    SELECT arr_to_bytea(ARRAY[1::float4, 2::float4, 3::float4, 4::float4]) AS x,
           arr_to_bytea(ARRAY[1.1::float4, 0.9::float4, 1.05::float4, 0.95::float4]) AS gamma,
           arr_to_bytea(ARRAY[0.1::float4, -0.2::float4, 0.3::float4, -0.4::float4]) AS beta
),
forward AS (
    SELECT pg_llm_layernorm(x, gamma, beta, 1e-5) AS y FROM fixture
),
backprop AS (
    SELECT llm_backprop((SELECT output FROM llm_tape ORDER BY id DESC LIMIT 1), 'test-model')
),
node AS (
    SELECT * FROM llm_tape ORDER BY id DESC LIMIT 1
),
ref AS (
    SELECT *
    FROM pg_llm_layernorm_backward(
        (SELECT data FROM llm_tensor_rt WHERE id = node.inputs[1]),
        (SELECT grad FROM llm_tensor_rt WHERE id = node.output),
        (SELECT data FROM llm_tensor_rt WHERE id = node.inputs[2]),
        (node.extra->>'eps')::FLOAT4)
)
SELECT 'dx matches' AS label,
       (SELECT grad FROM llm_tensor_rt WHERE id = node.inputs[1]) = ref.dx AS matches
FROM node, ref
UNION ALL
SELECT 'dgamma matches' AS label,
       (SELECT grad FROM llm_tensor_rt WHERE id = node.inputs[2]) = ref.dgamma AS matches
FROM node, ref
UNION ALL
SELECT 'dbeta matches' AS label,
       (SELECT grad FROM llm_tensor_rt WHERE id = node.inputs[3]) = ref.dbeta AS matches
FROM node, ref;

\echo 'attention backward'
TRUNCATE llm_tape;
TRUNCATE llm_tensor_rt;
TRUNCATE llm_tensor_map;
WITH params AS (
    SELECT 2 AS T, 4 AS D, 2 AS n_head,
           arr_to_bytea(ARRAY(SELECT (i::float4)/10 FROM generate_series(1, 8))) AS x,
           arr_to_bytea(ARRAY(SELECT (i::float4)/20 FROM generate_series(1, 48))) AS w_qkv,
           arr_to_bytea(ARRAY(SELECT (i::float4)/30 FROM generate_series(1, 12))) AS b_qkv,
           arr_to_bytea(ARRAY(SELECT (i::float4)/40 FROM generate_series(1, 16))) AS w_o,
           arr_to_bytea(ARRAY(SELECT (i::float4)/50 FROM generate_series(1, 4))) AS b_o
),
forward AS (
    SELECT pg_llm_attention(x, w_qkv, b_qkv, w_o, b_o, n_head, T, D) AS y
    FROM params
),
backprop AS (
    SELECT llm_backprop((SELECT output FROM llm_tape ORDER BY id DESC LIMIT 1), 'test-model')
),
node AS (
    SELECT * FROM llm_tape ORDER BY id DESC LIMIT 1
),
ref AS (
    SELECT *
    FROM pg_llm_attention_backward(
        (SELECT data FROM llm_tensor_rt WHERE id = node.inputs[1]),
        (SELECT data FROM llm_tensor_rt WHERE id = node.inputs[2]),
        (SELECT data FROM llm_tensor_rt WHERE id = node.inputs[3]),
        (SELECT data FROM llm_tensor_rt WHERE id = node.inputs[4]),
        (SELECT data FROM llm_tensor_rt WHERE id = node.inputs[5]),
        (SELECT grad FROM llm_tensor_rt WHERE id = node.output),
        (node.extra->>'n_head')::INT,
        (node.extra->>'T')::INT,
        (node.extra->>'D')::INT)
)
SELECT 'dx matches' AS label,
       (SELECT grad FROM llm_tensor_rt WHERE id = node.inputs[1]) = ref.dx AS matches
FROM node, ref
UNION ALL
SELECT 'dw_qkv matches' AS label,
       (SELECT grad FROM llm_tensor_rt WHERE id = node.inputs[2]) = ref.dw_qkv AS matches
FROM node, ref
UNION ALL
SELECT 'db_qkv matches' AS label,
       (SELECT grad FROM llm_tensor_rt WHERE id = node.inputs[3]) = ref.db_qkv AS matches
FROM node, ref
UNION ALL
SELECT 'dw_o matches' AS label,
       (SELECT grad FROM llm_tensor_rt WHERE id = node.inputs[4]) = ref.dw_o AS matches
FROM node, ref
UNION ALL
SELECT 'db_o matches' AS label,
       (SELECT grad FROM llm_tensor_rt WHERE id = node.inputs[5]) = ref.db_o AS matches
FROM node, ref;

\echo 'gelu backward'
TRUNCATE llm_tape;
TRUNCATE llm_tensor_rt;
TRUNCATE llm_tensor_map;
WITH fixture AS (
    SELECT arr_to_bytea(ARRAY[-1.5::float4, -0.5::float4, 0.25::float4, 1.75::float4]) AS x
),
forward AS (
    SELECT pg_llm_gelu(x) AS y FROM fixture
),
backprop AS (
    SELECT llm_backprop((SELECT output FROM llm_tape ORDER BY id DESC LIMIT 1), 'test-model')
),
node AS (
    SELECT * FROM llm_tape ORDER BY id DESC LIMIT 1
)
SELECT 'gelu grad matches' AS label,
       (SELECT grad FROM llm_tensor_rt WHERE id = node.inputs[1]) =
       pg_llm_gelu_backward(
           (SELECT data FROM llm_tensor_rt WHERE id = node.inputs[1]),
           (SELECT grad FROM llm_tensor_rt WHERE id = node.output)) AS matches
FROM node;

\echo 'transpose backward'
TRUNCATE llm_tape;
TRUNCATE llm_tensor_rt;
TRUNCATE llm_tensor_map;
WITH fixture AS (
    SELECT arr_to_bytea(ARRAY[1::float4, 2::float4, 3::float4, 4::float4, 5::float4, 6::float4]) AS x
),
forward AS (
    SELECT pg_llm_transpose(x, 2, 3) AS y FROM fixture
),
backprop AS (
    SELECT llm_backprop((SELECT output FROM llm_tape ORDER BY id DESC LIMIT 1), 'test-model')
),
node AS (
    SELECT * FROM llm_tape ORDER BY id DESC LIMIT 1
)
SELECT 'transpose grad matches' AS label,
       (SELECT grad FROM llm_tensor_rt WHERE id = node.inputs[1]) =
       pg_llm_transpose(
           (SELECT grad FROM llm_tensor_rt WHERE id = node.output),
           (node.extra->>'cols')::INT,
           (node.extra->>'rows')::INT) AS matches
FROM node;

\echo 'mlp backward'
TRUNCATE llm_tape;
TRUNCATE llm_tensor_rt;
TRUNCATE llm_tensor_map;
WITH params AS (
    SELECT 2 AS T, 4 AS D,
           arr_to_bytea(ARRAY(SELECT (i::float4)/10 FROM generate_series(1, 8))) AS x,
           arr_to_bytea(ARRAY(SELECT (i::float4)/30 FROM generate_series(1, 64))) AS w_fc,
           arr_to_bytea(ARRAY(SELECT (i::float4)/40 FROM generate_series(1, 8))) AS b_fc,
           arr_to_bytea(ARRAY(SELECT (i::float4)/50 FROM generate_series(1, 16))) AS w_proj,
           arr_to_bytea(ARRAY(SELECT (i::float4)/60 FROM generate_series(1, 4))) AS b_proj
),
forward_fc AS (
    SELECT pg_llm_matmul(x, w_fc, 2, 4, 8) AS hidden
    FROM params
),
post_fc AS (
    SELECT pg_llm_add(hidden, b_fc) AS pre_act FROM forward_fc, params
),
activated AS (
    SELECT pg_llm_gelu(pre_act) AS act FROM post_fc
),
projected AS (
    SELECT pg_llm_matmul(act, w_proj, 2, 8, 4) AS proj FROM activated, params
),
out AS (
    SELECT pg_llm_add(proj, b_proj) AS y FROM projected, params
),
backprop AS (
    SELECT llm_backprop((SELECT output FROM llm_tape ORDER BY id DESC LIMIT 1), 'test-model')
),
node AS (
    SELECT * FROM llm_tape ORDER BY id DESC LIMIT 1
),
ref AS (
    SELECT *
    FROM pg_llm_mlp_backward(
        (SELECT data FROM llm_tensor_rt WHERE id = node.inputs[1]),
        (SELECT data FROM llm_tensor_rt WHERE id = node.inputs[2]),
        (SELECT data FROM llm_tensor_rt WHERE id = node.inputs[3]),
        (SELECT data FROM llm_tensor_rt WHERE id = node.inputs[4]),
        (SELECT data FROM llm_tensor_rt WHERE id = node.inputs[5]),
        (SELECT grad FROM llm_tensor_rt WHERE id = node.output),
        (node.extra->>'T')::INT,
        (node.extra->>'D')::INT)
)
SELECT 'dx matches' AS label,
       (SELECT grad FROM llm_tensor_rt WHERE id = node.inputs[1]) = ref.dx AS matches
FROM node, ref
UNION ALL
SELECT 'dw_fc matches' AS label,
       (SELECT grad FROM llm_tensor_rt WHERE id = node.inputs[2]) = ref.dw_fc AS matches
FROM node, ref
UNION ALL
SELECT 'db_fc matches' AS label,
       (SELECT grad FROM llm_tensor_rt WHERE id = node.inputs[3]) = ref.db_fc AS matches
FROM node, ref
UNION ALL
SELECT 'dw_proj matches' AS label,
       (SELECT grad FROM llm_tensor_rt WHERE id = node.inputs[4]) = ref.dw_proj AS matches
FROM node, ref
UNION ALL
SELECT 'db_proj matches' AS label,
       (SELECT grad FROM llm_tensor_rt WHERE id = node.inputs[5]) = ref.db_proj AS matches
FROM node, ref;
