-- Verify AdamW optimizer matches decoupled formulation (Loshchilov & Hutter, 2019).
SET extra_float_digits = 3;

DROP TYPE IF EXISTS adamw_state CASCADE;
CREATE TYPE adamw_state AS (weight BYTEA, m BYTEA, v BYTEA);

DROP FUNCTION IF EXISTS pg_llm_adamw_step(
    BYTEA, BYTEA, BYTEA, BYTEA,
    REAL, REAL, REAL, REAL, REAL, INTEGER);
CREATE FUNCTION pg_llm_adamw_step(
    weight BYTEA, grad BYTEA, m BYTEA, v BYTEA,
    lr REAL, beta1 REAL, beta2 REAL, eps REAL,
    weight_decay REAL, step INTEGER)
RETURNS adamw_state
AS '/workspace/pg_gpt2/pg_llm_optim', 'pg_llm_adamw_step'
LANGUAGE C STRICT;

DROP TABLE IF EXISTS adamw_context;
CREATE TEMP TABLE adamw_context AS
WITH params AS (
    SELECT
        ARRAY[0.5::real, -1.0::real, 0.75::real] AS w,
        ARRAY[0.1::real, -0.2::real, 0.05::real] AS g,
        ARRAY[0.01::real, -0.02::real, 0.03::real] AS m,
        ARRAY[0.4::real, 0.5::real, 0.6::real] AS v,
        0.001::real AS lr,
        0.9::real AS b1,
        0.999::real AS b2,
        1e-8::real AS eps,
        0.01::real AS wd,
        5 AS step
),
inputs AS (
    SELECT
        (SELECT string_agg(le_bytes, ''::bytea ORDER BY ord)
         FROM unnest(w) WITH ORDINALITY AS t(val, ord)
         CROSS JOIN LATERAL (
             SELECT set_byte(
                        set_byte(
                            set_byte(
                                set_byte('\x00000000'::bytea, 0, get_byte(be, 3)),
                                1, get_byte(be, 2)),
                            2, get_byte(be, 1)),
                        3, get_byte(be, 0)) AS le_bytes
             FROM (SELECT pg_catalog.float4send(val) AS be) s
         )) AS weight,
        (SELECT string_agg(le_bytes, ''::bytea ORDER BY ord)
         FROM unnest(g) WITH ORDINALITY AS t(val, ord)
         CROSS JOIN LATERAL (
             SELECT set_byte(
                        set_byte(
                            set_byte(
                                set_byte('\x00000000'::bytea, 0, get_byte(be, 3)),
                                1, get_byte(be, 2)),
                            2, get_byte(be, 1)),
                        3, get_byte(be, 0)) AS le_bytes
             FROM (SELECT pg_catalog.float4send(val) AS be) s
         )) AS grad,
        (SELECT string_agg(le_bytes, ''::bytea ORDER BY ord)
         FROM unnest(m) WITH ORDINALITY AS t(val, ord)
         CROSS JOIN LATERAL (
             SELECT set_byte(
                        set_byte(
                            set_byte(
                                set_byte('\x00000000'::bytea, 0, get_byte(be, 3)),
                                1, get_byte(be, 2)),
                            2, get_byte(be, 1)),
                        3, get_byte(be, 0)) AS le_bytes
             FROM (SELECT pg_catalog.float4send(val) AS be) s
         )) AS m,
        (SELECT string_agg(le_bytes, ''::bytea ORDER BY ord)
         FROM unnest(v) WITH ORDINALITY AS t(val, ord)
         CROSS JOIN LATERAL (
             SELECT set_byte(
                        set_byte(
                            set_byte(
                                set_byte('\x00000000'::bytea, 0, get_byte(be, 3)),
                                1, get_byte(be, 2)),
                            2, get_byte(be, 1)),
                        3, get_byte(be, 0)) AS le_bytes
             FROM (SELECT pg_catalog.float4send(val) AS be) s
         )) AS v,
        lr, b1, b2, eps, wd, step,
        (1::real - power(b1::double precision, step)::real) AS bc1,
        (1::real - power(b2::double precision, step)::real) AS bc2
    FROM params
),
adamw AS (
    SELECT (pg_llm_adamw_step(weight, grad, m, v, lr, b1, b2, eps, wd, step)).*
    FROM inputs
),
reference_vals AS (
    SELECT ord,
           (inputs.b1 * params.m[ord] + (1::real - inputs.b1) * params.g[ord])::real AS m_new,
           (inputs.b2 * params.v[ord] + (1::real - inputs.b2) * params.g[ord] * params.g[ord])::real AS v_new,
           (
               params.w[ord]
               - inputs.lr * ((inputs.b1 * params.m[ord] + (1::real - inputs.b1) * params.g[ord])::real / inputs.bc1)
                 / ((sqrt((inputs.b2 * params.v[ord] + (1::real - inputs.b2) * params.g[ord] * params.g[ord]) / inputs.bc2))::real + inputs.eps)
               - inputs.lr * inputs.wd * params.w[ord]
           )::real AS w_new
    FROM params, inputs, generate_subscripts(params.w, 1) AS ord
),
reference AS (
    SELECT
        array_agg(w_new ORDER BY ord)::real[] AS weight_arr,
        array_agg(m_new ORDER BY ord)::real[] AS m_arr,
        array_agg(v_new ORDER BY ord)::real[] AS v_arr
    FROM reference_vals
),
reference_bytes AS (
    SELECT
        (SELECT string_agg(le_bytes, ''::bytea ORDER BY ord)
         FROM unnest(weight_arr) WITH ORDINALITY AS t(val, ord)
         CROSS JOIN LATERAL (
             SELECT set_byte(
                        set_byte(
                            set_byte(
                                set_byte('\x00000000'::bytea, 0, get_byte(be, 3)),
                                1, get_byte(be, 2)),
                            2, get_byte(be, 1)),
                        3, get_byte(be, 0)) AS le_bytes
             FROM (SELECT pg_catalog.float4send(val) AS be) s
         )) AS weight,
        (SELECT string_agg(le_bytes, ''::bytea ORDER BY ord)
         FROM unnest(m_arr) WITH ORDINALITY AS t(val, ord)
         CROSS JOIN LATERAL (
             SELECT set_byte(
                        set_byte(
                            set_byte(
                                set_byte('\x00000000'::bytea, 0, get_byte(be, 3)),
                                1, get_byte(be, 2)),
                            2, get_byte(be, 1)),
                        3, get_byte(be, 0)) AS le_bytes
             FROM (SELECT pg_catalog.float4send(val) AS be) s
         )) AS m,
        (SELECT string_agg(le_bytes, ''::bytea ORDER BY ord)
         FROM unnest(v_arr) WITH ORDINALITY AS t(val, ord)
         CROSS JOIN LATERAL (
             SELECT set_byte(
                        set_byte(
                            set_byte(
                                set_byte('\x00000000'::bytea, 0, get_byte(be, 3)),
                                1, get_byte(be, 2)),
                            2, get_byte(be, 1)),
                        3, get_byte(be, 0)) AS le_bytes
             FROM (SELECT pg_catalog.float4send(val) AS be) s
         )) AS v
    FROM reference
)
SELECT
    adamw.weight AS weight,
    adamw.m AS m,
    adamw.v AS v,
    reference.weight_arr,
    reference.m_arr,
    reference.v_arr,
    reference_bytes.weight AS ref_weight,
    reference_bytes.m AS ref_m,
    reference_bytes.v AS ref_v
FROM adamw, reference, reference_bytes;

SELECT 'adamw reference arrays' AS label,
       weight_arr AS weight,
       m_arr AS m,
       v_arr AS v
FROM adamw_context;

SELECT 'adamw hex match' AS label,
       encode(weight, 'hex') AS impl_weight_hex,
       encode(ref_weight, 'hex') AS ref_weight_hex,
       encode(m, 'hex') AS impl_m_hex,
       encode(ref_m, 'hex') AS ref_m_hex,
       encode(v, 'hex') AS impl_v_hex,
       encode(ref_v, 'hex') AS ref_v_hex
FROM adamw_context;

SELECT 'adamw equality' AS label,
       weight = ref_weight AS weight_equal,
       m = ref_m AS m_equal,
       v = ref_v AS v_equal
FROM adamw_context;
