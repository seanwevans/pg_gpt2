-- Comprehensive regression harness for pg_llm numerical kernels.
--
-- Each fixture is generated from PyTorch (float32) to ensure the
-- Postgres kernels stay numerically aligned with a trusted reference.
-- Forward kernels and their gradients are checked bit-for-bit by
-- comparing the bytea blobs emitted by pg_llm_* functions against the
-- PyTorch tensors encoded in hex.

SET extra_float_digits = 3;

-- Use CREATE OR REPLACE so the regression harness can be re-run idempotently
-- without requiring prior manual cleanup.
CREATE OR REPLACE FUNCTION pg_llm_gelu_backward(x BYTEA, dy BYTEA)
RETURNS BYTEA
AS '/workspace/pg_gpt2/pg_llm', 'pg_llm_gelu_backward'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION pg_llm_softmax_backward(y BYTEA, dy BYTEA)
RETURNS BYTEA
AS '/workspace/pg_gpt2/pg_llm', 'pg_llm_softmax_backward'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION pg_llm_layernorm_backward(x BYTEA, dy BYTEA, gamma BYTEA, eps REAL)
RETURNS RECORD
AS '/workspace/pg_gpt2/pg_llm', 'pg_llm_layernorm_backward'
LANGUAGE C STRICT;

-- Matmul -----------------------------------------------------------------
WITH fixture AS (
    SELECT
        decode('5c3fc53f503c96be49710bc0b684113fa1d18abf2d05b3bf', 'hex')::bytea AS a,
        decode('7a83ce3ee588563f442138bf0b83cebe18bd18bfc6673a3e', 'hex')::bytea AS b,
        decode('b17b084034ae813fe500ec3f56c1283f', 'hex')::bytea AS expected
)
SELECT 'matmul forward' AS label,
       encode(actual, 'hex') AS actual_hex,
       encode(expected, 'hex') AS expected_hex,
       actual = expected AS matches
FROM fixture,
LATERAL (
    SELECT pg_llm_matmul(a, b, 2, 3, 2) AS actual
) AS run;

-- Add --------------------------------------------------------------------
WITH fixture AS (
    SELECT
        decode('0000c0bf000000400000403f000080be', 'hex')::bytea AS a,
        decode('0000803e000040c0000090400000a03f', 'hex')::bytea AS b,
        decode('0000a0bf000080bf0000a8400000803f', 'hex')::bytea AS expected
)
SELECT 'add forward' AS label,
       encode(actual, 'hex') AS actual_hex,
       encode(expected, 'hex') AS expected_hex,
       actual = expected AS matches
FROM fixture,
LATERAL (
    SELECT pg_llm_add(a, b) AS actual
) AS run;

-- GELU -------------------------------------------------------------------
WITH fixture AS (
    SELECT
        decode('000080bf000000bf000000000000003f0000803f', 'hex')::bytea AS x,
        decode('867622bea2f81dbe00000000af03b13e5e62573f', 'hex')::bytea AS expected_y,
        decode('0000803f000000bf0000803e0000c0bf00000040', 'hex')::bytea AS dy,
        decode('48a1aabd5eaf87bd0000003e1e8fa6bf14aa0a40', 'hex')::bytea AS expected_dx
)
SELECT 'gelu forward' AS label,
       encode(actual, 'hex') AS actual_hex,
       encode(expected_y, 'hex') AS expected_hex,
       actual = expected_y AS matches
FROM fixture,
LATERAL (
    SELECT pg_llm_gelu(x) AS actual
) AS run;

WITH fixture AS (
    SELECT
        decode('000080bf000000bf000000000000003f0000803f', 'hex')::bytea AS x,
        decode('0000803f000000bf0000803e0000c0bf00000040', 'hex')::bytea AS dy,
        decode('48a1aabd5eaf87bd0000003e1e8fa6bf14aa0a40', 'hex')::bytea AS expected_dx
)
SELECT 'gelu backward' AS label,
       encode(actual, 'hex') AS actual_hex,
       encode(expected_dx, 'hex') AS expected_hex,
       actual = expected_dx AS matches
FROM fixture,
LATERAL (
    SELECT pg_llm_gelu_backward(x, dy) AS actual
) AS run;

-- Softmax ----------------------------------------------------------------
WITH fixture AS (
    SELECT
        decode('00000040000080bf0000003f00004040', 'hex')::bytea AS x,
        decode('4248803ea5604c3c36fd643d895a2e3f', 'hex')::bytea AS expected_y,
        decode('cdcccc3dcdcc4cbecdcc4c3d295c8f3d', 'hex')::bytea AS dy,
        decode('196cdd3b82345fbb38c2a8ba914207bb', 'hex')::bytea AS expected_dx
)
SELECT 'softmax forward' AS label,
       encode(actual, 'hex') AS actual_hex,
       encode(expected_y, 'hex') AS expected_hex,
       actual = expected_y AS matches
FROM fixture,
LATERAL (
    SELECT pg_llm_softmax(x) AS actual
) AS run;

WITH fixture AS (
    SELECT
        decode('4248803ea5604c3c36fd643d895a2e3f', 'hex')::bytea AS y,
        decode('cdcccc3dcdcc4cbecdcc4c3d295c8f3d', 'hex')::bytea AS dy,
        decode('196cdd3b82345fbb38c2a8ba914207bb', 'hex')::bytea AS expected_dx
)
SELECT 'softmax backward' AS label,
       encode(actual, 'hex') AS actual_hex,
       encode(expected_dx, 'hex') AS expected_hex,
       actual = expected_dx AS matches
FROM fixture,
LATERAL (
    SELECT pg_llm_softmax_backward(y, dy) AS actual
) AS run;

-- LayerNorm --------------------------------------------------------------
WITH fixture AS (
    SELECT
        decode('0000803e000040bf0000803f0000003f', 'hex')::bytea AS x,
        decode('0000c03f000000bf000000400000403f', 'hex')::bytea AS gamma,
        decode('cdcccc3dcdcc4cbe9a99993ecdccccbe', 'hex')::bytea AS beta,
        decode('cdcccc3dd69e153fbad0294019bdd8bd', 'hex')::bytea AS expected_y,
        decode('000000bf0000803e0000803f0000a0bf', 'hex')::bytea AS dy,
        decode('6007a0bff055893f59ab034042a5f0bf', 'hex')::bytea AS expected_dx,
        decode('0000000009d2c8be879d963f8b06fbbe', 'hex')::bytea AS expected_dgamma,
        decode('000000bf0000803e0000803f0000a0bf', 'hex')::bytea AS expected_dbeta,
        1e-5::real AS eps
)
SELECT 'layernorm forward' AS label,
       encode(actual, 'hex') AS actual_hex,
       encode(expected_y, 'hex') AS expected_hex,
       actual = expected_y AS matches
FROM fixture,
LATERAL (
    SELECT pg_llm_layernorm(x, gamma, beta, eps) AS actual
) AS run;

WITH fixture AS (
    SELECT
        decode('0000803e000040bf0000803f0000003f', 'hex')::bytea AS x,
        decode('000000bf0000803e0000803f0000a0bf', 'hex')::bytea AS dy,
        decode('0000c03f000000bf000000400000403f', 'hex')::bytea AS gamma,
        decode('6007a0bff055893f59ab034042a5f0bf', 'hex')::bytea AS expected_dx,
        decode('0000000009d2c8be879d963f8b06fbbe', 'hex')::bytea AS expected_dgamma,
        decode('000000bf0000803e0000803f0000a0bf', 'hex')::bytea AS expected_dbeta,
        1e-5::real AS eps
)
SELECT 'layernorm backward' AS label,
       encode(dx, 'hex') AS dx_hex,
       encode(expected_dx, 'hex') AS expected_dx_hex,
       encode(dgamma, 'hex') AS dgamma_hex,
       encode(expected_dgamma, 'hex') AS expected_dgamma_hex,
       encode(dbeta, 'hex') AS dbeta_hex,
       encode(expected_dbeta, 'hex') AS expected_dbeta_hex,
       dx = expected_dx AS dx_match,
       dgamma = expected_dgamma AS dgamma_match,
       dbeta = expected_dbeta AS dbeta_match
FROM fixture,
LATERAL (
    SELECT (pg_llm_layernorm_backward(x, dy, gamma, eps)).*
) AS run(dx, dgamma, dbeta);

-- Cross entropy ----------------------------------------------------------
WITH fixture AS (
    SELECT
        decode('cdcc4c3e0000c0bf000040400000403f', 'hex')::bytea AS logits,
        2 AS target,
        decode('3128273e', 'hex')::bytea AS expected
)
SELECT 'cross entropy' AS label,
       encode(actual, 'hex') AS actual_hex,
       encode(expected, 'hex') AS expected_hex,
       actual = expected AS matches
FROM fixture,
LATERAL (
    SELECT pg_catalog.float4send(pg_llm_cross_entropy(logits, target)) AS actual
) AS run;

-- Dropout (inference path is deterministic) ------------------------------
WITH fixture AS (
    SELECT
        decode('0000003f000080bf0000204000000000', 'hex')::bytea AS input,
        0.3::real AS p,
        decode('0000003f000080bf0000204000000000', 'hex')::bytea AS expected
)
SELECT 'dropout eval forward' AS label,
       encode(actual, 'hex') AS actual_hex,
       encode(expected, 'hex') AS expected_hex,
       actual = expected AS matches
FROM fixture,
LATERAL (
    SELECT pg_llm_dropout(input, p, false) AS actual
) AS run;

-- Attention --------------------------------------------------------------
WITH fixture AS (
    SELECT
        decode('5151fe3e28950dbeebce253fa4f2c23fe7c56fbe99c16fbea523ca3f9a76443f', 'hex')::bytea AS x,
        decode('f25ef0be37e50a3f1545edbe2174eebef5c4773e5ee6f4bf1ccadcbf13f20fbf73a481bf07e5a03e447468bf5ec6b4bf619abb3fe73167be
3a4c8a3d265eb6bfab5c0bbf622be33dc25393bf7e5bc03e75c319bfe25895be72091abf7417ed3f77235dbc126387bf4e92523f9b449cbf57e0553e78d6fabf
0002aabffe95493e250c3d3f2b7b2f3e02d9ecbd432a9abe3540bdbfb64738bfd8d8ebbec84f873fbdeeaf3e4dabe1bf56eea53e8129c5bec34a2dbfd1961c3f
cbf7833f60686e3f', 'hex')::bytea AS w_qkv,
        decode('0ad7233c0ad7a3bc8fc2f53c0ad723bdcdcc4c3d8fc275bd295c8f3d0ad7a3bdec51b83dcdccccbdae47e13d8fc2f5bd', 'hex')::bytea AS b_qkv,
        decode('f6d656bf16519ebe5c9ba93e53bd793f5356f5be631d3ebe629c8dbf4c1d99bfb101503f4699ad3f0b7a93bdc473803f5b28b93e912625bf
 d808b93e62dec43f', 'hex')::bytea AS w_o,
        decode('0ad7a33b8fc275bccdcccc3c295c0fbd', 'hex')::bytea AS b_o,
        decode('0a0da13fa9bed03fe230c8bfa7e84abfafd9603f9a68bb3fdae9a7bf3ab6bfbe', 'hex')::bytea AS expected
)
SELECT 'attention forward' AS label,
       encode(actual, 'hex') AS actual_hex,
       encode(expected, 'hex') AS expected_hex,
       actual = expected AS matches
FROM fixture,
LATERAL (
    SELECT pg_llm_attention(x, w_qkv, b_qkv, w_o, b_o, 2, 2, 4) AS actual
) AS run;
