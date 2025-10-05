DROP FUNCTION IF EXISTS pg_llm_dropout(BYTEA, REAL, BOOLEAN);
CREATE FUNCTION pg_llm_dropout(input BYTEA, p REAL, training BOOLEAN)
RETURNS BYTEA
AS '/workspace/pg_gpt2/pg_llm', 'pg_llm_dropout'
LANGUAGE C STRICT;

SET extra_float_digits = 3;

WITH ones AS (
    SELECT string_agg(le_bytes, ''::bytea ORDER BY ord) AS input
    FROM generate_series(1, 8) WITH ORDINALITY AS gs(_, ord)
    CROSS JOIN LATERAL (
        SELECT set_byte(
                   set_byte(
                       set_byte(
                           set_byte('\x00000000'::bytea, 0, get_byte(be, 3)),
                           1, get_byte(be, 2)),
                       2, get_byte(be, 1)),
                   3, get_byte(be, 0)) AS le_bytes
        FROM (SELECT pg_catalog.float4send(1.0::float4) AS be) s
    )
)
SELECT 'dropout inference passthrough' AS label,
       encode(input, 'hex') AS input_hex,
       encode(out_bytes, 'hex') AS output_hex,
       out_bytes = input AS exact_match
FROM ones,
LATERAL (
    SELECT pg_llm_dropout(input, 0.1::float4, false) AS out_bytes
) d;

SELECT setseed(0.5);

WITH ones AS (
    SELECT string_agg(le_bytes, ''::bytea ORDER BY ord) AS input
    FROM generate_series(1, 8) WITH ORDINALITY AS gs(_, ord)
    CROSS JOIN LATERAL (
        SELECT set_byte(
                   set_byte(
                       set_byte(
                           set_byte('\x00000000'::bytea, 0, get_byte(be, 3)),
                           1, get_byte(be, 2)),
                       2, get_byte(be, 1)),
                   3, get_byte(be, 0)) AS le_bytes
        FROM (SELECT pg_catalog.float4send(1.0::float4) AS be) s
    )
)
SELECT 'dropout training sample' AS label,
       encode(pg_llm_dropout(input, 0.1::float4, true), 'hex') AS output_hex
FROM ones;

SELECT setseed(0.5);

WITH ones AS (
    SELECT string_agg(le_bytes, ''::bytea ORDER BY ord) AS input
    FROM generate_series(1, 256) WITH ORDINALITY AS gs(_, ord)
    CROSS JOIN LATERAL (
        SELECT set_byte(
                   set_byte(
                       set_byte(
                           set_byte('\x00000000'::bytea, 0, get_byte(be, 3)),
                           1, get_byte(be, 2)),
                       2, get_byte(be, 1)),
                   3, get_byte(be, 0)) AS le_bytes
        FROM (SELECT pg_catalog.float4send(1.0::float4) AS be) s
    )
),
 drop_samples AS (
    SELECT pg_llm_dropout(input, 0.1::float4, true) AS out_bytes
    FROM ones
 ),
 chunks AS (
    SELECT c.chunk
    FROM drop_samples,
         LATERAL generate_series(0, (octet_length(out_bytes) / 4) - 1) AS g(idx)
         CROSS JOIN LATERAL (
             SELECT substring(out_bytes FROM g.idx * 4 + 1 FOR 4) AS chunk
         ) AS c
 ),
 scale_bytes AS (
    SELECT set_byte(
               set_byte(
                   set_byte(
                       set_byte('\x00000000'::bytea, 0, get_byte(be, 3)),
                       1, get_byte(be, 2)),
                   2, get_byte(be, 1)),
               3, get_byte(be, 0)) AS scaled_chunk
    FROM (SELECT pg_catalog.float4send((1.0 / (1.0 - 0.1))::float4) AS be) s
),
 stats AS (
    SELECT
        SUM(CASE WHEN chunk = '\x00000000'::bytea THEN 1 ELSE 0 END) AS zero_count,
        SUM(CASE WHEN chunk = scale_bytes.scaled_chunk THEN 1 ELSE 0 END) AS scaled_count,
        COUNT(*) AS total_count
    FROM chunks, scale_bytes
)
SELECT 'dropout training stats' AS label,
       ROUND((scaled_count::numeric * (1.0 / (1.0 - 0.1))) / total_count, 6) AS mean_val,
        ABS((scaled_count::numeric * (1.0 / (1.0 - 0.1))) / total_count - 1.0) < 0.05 AS approx_expected,
       zero_count,
       total_count,
       scaled_count = total_count - zero_count AS scaled_matches
FROM stats;
