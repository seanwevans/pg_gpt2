SET client_min_messages = warning;
SET extra_float_digits = 3;

DROP FUNCTION IF EXISTS pg_llm_sample(BYTEA, FLOAT4, INT, FLOAT4);
CREATE FUNCTION pg_llm_sample(
    logits BYTEA,
    temperature FLOAT4 DEFAULT 1.0,
    topk INT DEFAULT 50,
    topp FLOAT4 DEFAULT 0.95)
RETURNS INT
AS '/workspace/pg_gpt2/pg_llm', 'pg_llm_sample'
LANGUAGE C STRICT;

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

SELECT setseed(0.4242);

SELECT 'baseline sample' AS label,
       pg_llm_sample(arr_to_bytea(ARRAY[-1::float4, 0::float4, 2::float4, 1.5::float4]),
                     1.0, 0, 1.0) AS token_id;

SELECT 'topk=1 returns argmax' AS label,
       pg_llm_sample(arr_to_bytea(ARRAY[0::float4, -0.5::float4, 3::float4, -1::float4]),
                     1.0, 1, 1.0) AS token_id;

SELECT setseed(0.4242);
SELECT 'topp=0.2 prunes tail' AS label,
       pg_llm_sample(arr_to_bytea(ARRAY[0::float4, 1::float4, 2::float4, 3::float4]),
                     1.0, 0, 0.2) AS token_id;

SELECT setseed(0.4242);
SELECT 'temperature=0.5 skews distribution' AS label,
       pg_llm_sample(arr_to_bytea(ARRAY[-0.25::float4, -0.24::float4, -0.20::float4, -0.23::float4]),
                     0.5, 0, 1.0) AS token_id;
