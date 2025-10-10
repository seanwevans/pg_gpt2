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

CREATE OR REPLACE FUNCTION pg_llm_dropout_backward(
    input BYTEA,
    output BYTEA,
    grad BYTEA,
    p REAL,
    training BOOLEAN)
RETURNS BYTEA
AS '/workspace/pg_gpt2/pg_llm', 'pg_llm_dropout_backward'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION pg_llm_layernorm_backward(x BYTEA, dy BYTEA, gamma BYTEA, eps REAL)
RETURNS RECORD
AS '/workspace/pg_gpt2/pg_llm', 'pg_llm_layernorm_backward'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION pg_llm_cross_entropy_backward(logits BYTEA, target INT)
RETURNS BYTEA
AS '/workspace/pg_gpt2/pg_llm', 'pg_llm_cross_entropy_backward'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION pg_llm_attention_backward(
    x BYTEA,
    w_qkv BYTEA,
    b_qkv BYTEA,
    w_o BYTEA,
    b_o BYTEA,
    dy BYTEA,
    n_head INT,
    T INT,
    D INT)
RETURNS RECORD
AS '/workspace/pg_gpt2/pg_llm', 'pg_llm_attention_backward'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION pg_llm_mlp_backward(
    x BYTEA,
    w_fc BYTEA,
    b_fc BYTEA,
    w_proj BYTEA,
    b_proj BYTEA,
    dy BYTEA,
    T INT,
    D INT)
RETURNS RECORD
AS '/workspace/pg_gpt2/pg_llm', 'pg_llm_mlp_backward'
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

-- MLP -------------------------------------------------------------------
WITH fixture AS (
    SELECT
        decode('df48e4bdd980f63dbc40bdbe213076bed13499bfb54a563e42ec78bfa94a41bf', 'hex')::bytea AS x,
        decode('c9bc9dbdd39e82bfe0022dbe9ff26a3f365dca3f7a88a63f123da33fa6c64dbe7231fe3e4441c9bf3c71773f62f692bfc85594bf23a4a63e56aa21bf76c235c03498a9bff2b5363e698f08c0a9b3863f41e7c6be4b316fbfd18fffbe6f178bbf5a6b613fabeec63fe969203febb233be0149c93dd380bfbd4c4d883ee3c415bf6478603f6aa2cf3f342dbdbfa609913f79329cbf5d2ca83f86d2863ff2220e3e47d40f4016bb4dbf95ca8fbe8109453fe6d828bffd444cbf433e3c3e0cda6a3ebabe033f4e677e3fb57884be0a918abfe2c935bdbbd0cf3feba914c0083e8b3f07eb2b3fcd7b313fa2df72bfd3af9cbdc24d1cbe0116ef3dd169e13e7028b9bf', 'hex')::bytea AS w_fc,
        decode('aeb5823e66b40cbf7b8a803f67c5533f6024cabe107cfa3eb9045ebe16a3dfbffc1ecdbffec689bfb934673fcaca38bf715718bf5a1236bfa07a1f3f98baafbf', 'hex')::bytea AS b_fc,
        decode('63c20dc026dfa8bf1fdb05c0757d763fee8002bdaf39f5be924d443ff904e13c4417ff3fee77af3fdf3800bf4cfd8ebe340504c06ce1d03b75537dbfa99c333f6ec670bf1ca6efbe2d1f843f95e590be1f4afc3e15a866bc98a08cbe809b43bf44c3b23f78b27ebf6d62cfba9da09f3ffee89dbd1481a33f23d5babfb4350ac0393584be619b02c0d1324dbf277c51bf444d97bf505393be73b31abf1da9193f9ddfb3bf229817bf6c7382be756c933fdb4a92bc8151da3e950744bf444a5fbde11aa3bfd10de73e297c69be1c216c3f218a523ef171febed001153f483f523e0b819abe10992bbf06fa1dbf545955bfc1bdf73e702a0abe31f4583ebf135fbf', 'hex')::bytea AS w_proj,
        decode('be26953ea29800bfa83c6ebf1699683f', 'hex')::bytea AS b_proj,
        decode('745091bf4a811c3f692294bea2322a3fdc77973f23e045bfe46721bf56f892bf', 'hex')::bytea AS dy,
        decode('307968404735f540bd559540b7d07fc027ed27c0d18721c1ea15813f8380053f', 'hex')::bytea AS expected_dx,
        decode('ef4c89befe8306bd3d4a0fc0cecdd93ea0e21fbece18023c5754313e3e308d3db8499cbbdda8fe3f00d92c40d6dbccbdd48e6840a9d9673b8ea9efbf9940f2bfbf4cf03d0c8d3b3c520a663e015b6b3e2cad1d3d1b0ad5bb1ae093bd2f5835bc5076ab3b279caabebf1a74bedd8d783c7c220cbfc37f0bbc1e11933e6046ac3ea054d3be63ee2dbd21beadbf82fcfdbe3a9b21beadbaa83c7b6b833e263c5a3d3a6f83bc63b3c93f1709c63f5d4999bdb95b2d40fd7eca3ce97cb4bf44c6c6bf706f93be4686f8bc2cc790bfa7837abe5862f0bd7b555f3ce242383e80532b3d93d22abcee7c9d3fda9aa73f037272bdeb360940b0b57e3c21868ebf69059abf', 'hex')::bytea AS expected_dw_fc,
        decode('767d5b3ff03ca13d18f3bd3e3b4a1840aaa66c3e177557bd7f8805bf04a647bd514c323d9ceec3bf70191ebeea77803d6ab60fc0e96d97bdbd0f9a3f6bcfd03f', 'hex')::bytea AS expected_db_fc,
        decode('477ccbbcb36c053d8d88003e06aca13d7c65463d306a6dbca556bc3da5582c3c8877b03f343d88bf0b0feebf8692f1bf11b33dbfa3e6d43eddb596bd3b01fb3ede19bebd20668e3d0377e23d6897f53d1d21423e31fed0bda215473df814e3bd05b777bd8e4b573d75acf93d3fe3d13d56b8d23c9e994dbcf0047e3cd5322fbce994813c268c0bbcd847843be8b917bc99bad83f37d88abfef9241bf872ac9bf5f3cc3be3022e23da9c63fbf26c1c2bd4e06173e49dc98bd293f903d510890bdb1b0ea3ff20c98bf2ef568bf9b89dfbf18a96b3b9221903cb34a0a3ef1d9813dac78e83ef1eeb1be758117bf03881cbfef9a5d40f85a0fc081fad8bf105852c0', 'hex')::bytea AS expected_dw_proj,
        decode('00ed443d647b25be18796bbf147cf7be', 'hex')::bytea AS expected_db_proj
)
SELECT 'mlp backward' AS label,
       encode(dx, 'hex') AS dx_hex,
       encode(expected_dx, 'hex') AS expected_dx_hex,
       encode(dw_fc, 'hex') AS dw_fc_hex,
       encode(expected_dw_fc, 'hex') AS expected_dw_fc_hex,
       encode(db_fc, 'hex') AS db_fc_hex,
       encode(expected_db_fc, 'hex') AS expected_db_fc_hex,
       encode(dw_proj, 'hex') AS dw_proj_hex,
       encode(expected_dw_proj, 'hex') AS expected_dw_proj_hex,
       encode(db_proj, 'hex') AS db_proj_hex,
       encode(expected_db_proj, 'hex') AS expected_db_proj_hex,
       dx = expected_dx AS dx_match,
       dw_fc = expected_dw_fc AS dw_fc_match,
       db_fc = expected_db_fc AS db_fc_match,
       dw_proj = expected_dw_proj AS dw_proj_match,
       db_proj = expected_db_proj AS db_proj_match
FROM fixture,
LATERAL (
    SELECT (pg_llm_mlp_backward(x, w_fc, b_fc, w_proj, b_proj, dy, 2, 4)).*
) AS run(dx, dw_fc, db_fc, dw_proj, db_proj);

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

WITH fixture AS (
    SELECT
        decode('000040c0000000bf000000000000003f00004040', 'hex')::bytea AS x,
        decode('c0b784bba2f81dbe00000000af03b13ea4bd3f40', 'hex')::bytea AS expected_y,
        decode('0000803e000040bf0000003f000080be0000003e', 'hex')::bytea AS dy,
        decode('80b743bb0d87cbbd0000803e28145ebe6f87013e', 'hex')::bytea AS expected_dx
)
SELECT 'gelu forward (wide range)' AS label,
       encode(actual, 'hex') AS actual_hex,
       encode(expected_y, 'hex') AS expected_hex,
       actual = expected_y AS matches
FROM fixture,
LATERAL (
    SELECT pg_llm_gelu(x) AS actual
) AS run;

WITH fixture AS (
    SELECT
        decode('000040c0000000bf000000000000003f00004040', 'hex')::bytea AS x,
        decode('0000803e000040bf0000003f000080be0000003e', 'hex')::bytea AS dy,
        decode('80b743bb0d87cbbd0000803e28145ebe6f87013e', 'hex')::bytea AS expected_dx
)
SELECT 'gelu backward (wide range)' AS label,
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

WITH fixture AS (
    SELECT
        decode('000000c0000000000000004000008040000080c0', 'hex')::bytea AS x,
        decode('11780c3bdebd813cc7aaef3d3d5d5d3f53159839', 'hex')::bytea AS expected_y,
        decode('cdcccc3dcdcc4cbe9a99993ecdccccbe0000003f', 'hex')::bytea AS dy,
        decode('c761683a16c8eb3a340e933dd50999bdec767739', 'hex')::bytea AS expected_dx
)
SELECT 'softmax forward (5d)' AS label,
       encode(actual, 'hex') AS actual_hex,
       encode(expected_y, 'hex') AS expected_hex,
       actual = expected_y AS matches
FROM fixture,
LATERAL (
    SELECT pg_llm_softmax(x) AS actual
) AS run;

WITH fixture AS (
    SELECT
        decode('11780c3bdebd813cc7aaef3d3d5d5d3f53159839', 'hex')::bytea AS y,
        decode('cdcccc3dcdcc4cbe9a99993ecdccccbe0000003f', 'hex')::bytea AS dy,
        decode('c761683a16c8eb3a340e933dd50999bdec767739', 'hex')::bytea AS expected_dx
)
SELECT 'softmax backward (5d)' AS label,
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

WITH fixture AS (
    SELECT
        decode('0000803f000080bf0000003f00000040000000bf00004040', 'hex')::bytea AS x,
        decode('0000003f000080bf0000c03f000000bf0000803e0000a0bf', 'hex')::bytea AS gamma,
        decode('cdcccc3dcdcc4cbe9a99993ecdccccbe0000003f9a9919bf', 'hex')::bytea AS beta,
        decode('cc7c243efb23913f50a682bd980d53bf36d3833ee48324c0', 'hex')::bytea AS expected_y,
        decode('000000bf0000803e000040bf0000803f0000a0bf0000003f', 'hex')::bytea AS dy,
        decode('1c034a3e2c4bd43db5efecbec0ba7e3d5818a73dd0de933c', 'hex')::bytea AS expected_dx,
        decode('965978bd95bdaabe2e433a3e624e593ffc379b3fc9c8493f', 'hex')::bytea AS expected_dgamma,
        decode('000000bf0000803e000040bf0000803f0000a0bf0000003f', 'hex')::bytea AS expected_dbeta,
        1e-4::real AS eps
)
SELECT 'layernorm forward (6d)' AS label,
       encode(actual, 'hex') AS actual_hex,
       encode(expected_y, 'hex') AS expected_hex,
       actual = expected_y AS matches
FROM fixture,
LATERAL (
    SELECT pg_llm_layernorm(x, gamma, beta, eps) AS actual
) AS run;

WITH fixture AS (
    SELECT
        decode('0000803f000080bf0000003f00000040000000bf00004040', 'hex')::bytea AS x,
        decode('000000bf0000803e000040bf0000803f0000a0bf0000003f', 'hex')::bytea AS dy,
        decode('0000003f000080bf0000c03f000000bf0000803e0000a0bf', 'hex')::bytea AS gamma,
        decode('1c034a3e2c4bd43db5efecbec0ba7e3d5818a73dd0de933c', 'hex')::bytea AS expected_dx,
        decode('965978bd95bdaabe2e433a3e624e593ffc379b3fc9c8493f', 'hex')::bytea AS expected_dgamma,
        decode('000000bf0000803e000040bf0000803f0000a0bf0000003f', 'hex')::bytea AS expected_dbeta,
        1e-4::real AS eps
)
SELECT 'layernorm backward (6d)' AS label,
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
        decode('3128273e', 'hex')::bytea AS expected,
        decode('5a90533dce981a3c0c3a1abecc58b73d', 'hex')::bytea AS expected_backward
)
SELECT 'cross entropy' AS label,
       encode(actual, 'hex') AS actual_hex,
       encode(expected, 'hex') AS expected_hex,
       actual = expected AS matches
FROM fixture,
LATERAL (
    SELECT pg_catalog.float4send(pg_llm_cross_entropy(logits, target)) AS actual
) AS run;

WITH fixture AS (
    SELECT
        decode('cdcc4c3e0000c0bf000040400000403f', 'hex')::bytea AS logits,
        2 AS target,
        decode('5a90533dce981a3c0c3a1abecc58b73d', 'hex')::bytea AS expected_dx
)
SELECT 'cross entropy backward' AS label,
       encode(actual, 'hex') AS actual_hex,
       encode(expected_dx, 'hex') AS expected_hex,
       actual = expected_dx AS matches
FROM fixture,
LATERAL (
    SELECT pg_llm_cross_entropy_backward(logits, target) AS actual
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

WITH fixture AS (
    SELECT
        decode('0000803f000000c00000000000004040', 'hex')::bytea AS x,
        decode('6edbb63f000000000000000000000000', 'hex')::bytea AS y,
        decode('0000003f0000c0bf00000040000000bf', 'hex')::bytea AS dy,
        0.3::real AS p,
        true AS training,
        decode('6edb363f000000006edb364000000000', 'hex')::bytea AS expected_dx
)
SELECT 'dropout backward' AS label,
       encode(actual, 'hex') AS actual_hex,
       encode(expected_dx, 'hex') AS expected_hex,
       actual = expected_dx AS matches
FROM fixture,
LATERAL (
    SELECT pg_llm_dropout_backward(x, y, dy, p, training) AS actual
) AS run;

WITH fixture AS (
    SELECT
        decode('cdcccc3dcdcc4cbe9a99993ecdccccbe', 'hex')::bytea AS input,
        0.0::real AS p,
        true AS training,
        decode('cdcccc3dcdcc4cbe9a99993ecdccccbe', 'hex')::bytea AS expected
)
SELECT 'dropout train zero-prob' AS label,
       encode(actual, 'hex') AS actual_hex,
       encode(expected, 'hex') AS expected_hex,
       actual = expected AS matches
FROM fixture,
LATERAL (
    SELECT pg_llm_dropout(input, p, training) AS actual
) AS run;

WITH fixture AS (
    SELECT
        decode('0000803f0000004000000000000040c0', 'hex')::bytea AS x,
        decode('0000004000000000000000000000c0c0', 'hex')::bytea AS y,
        decode('cdcc4c3ecdccccbe9a99193fcdcc4cbf', 'hex')::bytea AS dy,
        0.5::real AS p,
        true AS training,
        decode('cdcccc3e000000009a99993fcdccccbf', 'hex')::bytea AS expected_dx
)
SELECT 'dropout backward (mask)' AS label,
       encode(actual, 'hex') AS actual_hex,
       encode(expected_dx, 'hex') AS expected_hex,
       actual = expected_dx AS matches
FROM fixture,
LATERAL (
    SELECT pg_llm_dropout_backward(x, y, dy, p, training) AS actual
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

CREATE TEMP TABLE attention_fixture_ones (
    x BYTEA,
    w_qkv BYTEA,
    b_qkv BYTEA,
    w_o BYTEA,
    b_o BYTEA,
    dy BYTEA,
    expected_dx BYTEA,
    expected_dw_qkv BYTEA,
    expected_db_qkv BYTEA,
    expected_dw_o BYTEA,
    expected_db_o BYTEA
);

INSERT INTO attention_fixture_ones VALUES (
    decode('5151fe3e28950dbeebce253fa4f2c23fe7c56fbe99c16fbea523ca3f9a76443f', 'hex'),
    decode('f25ef0be37e50a3f1545edbe2174eebef5c4773e5ee6f4bf1ccadcbf13f20fbf73a481bf07e5a03e447468bf5ec6b4bf619abb3fe73167be3a4c8a3d265eb6bfab5c0bbf622be33dc25393bf7e5bc03e75c319bfe25895be72091abf7417ed3f77235dbc126387bf4e92523f9b449cbf57e0553e78d6fabf0002aabffe95493e250c3d3f2b7b2f3e02d9ecbd432a9abe3540bdbfb64738bfd8d8ebbec84f873fbdeeaf3e4dabe1bf56eea53e8129c5bec34a2dbfd1961c3fcbf7833f60686e3f', 'hex'),
    decode('0ad7233c0ad7a3bc8fc2f53c0ad723bdcdcc4c3d8fc275bd295c8f3d0ad7a3bdec51b83dcdccccbdae47e13d8fc2f5bd', 'hex'),
    decode('f6d656bf16519ebe5c9ba93e53bd793f5356f5be631d3ebe629c8dbf4c1d99bfb101503f4699ad3f0b7a93bdc473803f5b28b93e912625bfd808b93e62dec43f', 'hex'),
    decode('0ad7a33b8fc275bccdcccc3c295c0fbd', 'hex'),
    decode('0000803f0000803f0000803f0000803f0000803f0000803f0000803f0000803f', 'hex'),
    decode('66cd34c177d83a4005e44bc0e11d6040aaf569bf48f46a3fb440243ff784ee3f', 'hex'),
    decode('05c71e3c771f3bbdb2e74bbbbce0c23cd32b613e4422bd3e9a42bf3d77b67fbd3355073e6f791ec015cc0e40e7db943f2cc41e3c1b1c3bbd09e44bbb3cddc23c134cec3cab7a463df7b5483c8f2c06bcb2ea40bdafe7613f62f977bf194001bf3adb85bdbac09d3ea4e6ab3c664a24bed57e8fbe430ff1be19c5f3bd76f5a23d6bf1723e293e8ec0db20a3409e0d2a40061902bdc052193ef612273c6cad9fbd20ca683e7188c33e28bbc53da12e84bd87ade93e6cd108c1530f0941cbe08e40', 'hex'),
    decode('dc8529bd64c9473e60b4593c0411d0bd0000003400008034000080b200000032002ea23e77e9bdc0216bc64067d74e40', 'hex'),
    decode('505faabf505faabf505faabf505faabf58c60a4058c60a4058c60a4058c60a40766f1a40766f1a40766f1a40766f1a407cc6783e7cc6783e7cc6783e7cc6783e', 'hex'),
    decode('00000040000000400000004000000040', 'hex')
);

WITH fixture AS (
    SELECT
        decode('5151fe3e28950dbeebce253fa4f2c23fe7c56fbe99c16fbea523ca3f9a76443f', 'hex')::bytea AS x,
        decode('f25ef0be37e50a3f1545edbe2174eebef5c4773e5ee6f4bf1ccadcbf13f20fbf73a481bf07e5a03e447468bf5ec6b4bf619abb3fe73167be3a4c8a3d265eb6bfab5c0bbf622be33dc25393bf7e5bc03e75c319bfe25895be72091abf7417ed3f77235dbc126387bf4e92523f9b449cbf57e0553e78d6fabf0002aabffe95493e250c3d3f2b7b2f3e02d9ecbd432a9abe3540bdbfb64738bfd8d8ebbec84f873fbdeeaf3e4dabe1bf56eea53e8129c5bec34a2dbfd1961c3fcbf7833f60686e3f', 'hex')::bytea AS w_qkv,
        decode('0ad7233c0ad7a3bc8fc2f53c0ad723bdcdcc4c3d8fc275bd295c8f3d0ad7a3bdec51b83dcdccccbdae47e13d8fc2f5bd', 'hex')::bytea AS b_qkv,
        decode('f6d656bf16519ebe5c9ba93e53bd793f5356f5be631d3ebe629c8dbf4c1d99bfb101503f4699ad3f0b7a93bdc473803f5b28b93e912625bfd808b93e62dec43f', 'hex')::bytea AS w_o,
        decode('0ad7a33b8fc275bccdcccc3c295c0fbd', 'hex')::bytea AS b_o,
        decode('5c3fc53f503c96be49710bc0b684113fa1d18abf2d05b3bf7a83ce3ee588563f', 'hex')::bytea AS dy,
        decode('b56d17c0044c2c401398f83e42c9993e048d00c0852805403efef2be6669b93f', 'hex')::bytea AS expected_dx,
        decode('000000000000000044f9733ba32ce9bc00000000000000005c8ad9bcc7aa233dc8ffea3e45df3f3e516bbbbe7984f13e0000000000000000f3cb25ba8a769e3b00000000000000003a4a64bb0fc1ab3be6d302bef2a255bd4894f73cecd500bf0000000000000000b683173bead090bc0000000000000000ffa10a3de59950bd9036193fe4307a3e6a76d5bc803f3d400000000000000000ec2f1a3c235d93bd00000000000000007be6e0bc5c34293da123b43f7f14133f5f4744bfa7b45240', 'hex')::bytea AS expected_dw_qkv,
        decode('0000000000000000168bb43b0e8e2cbd00000000000000000000e03200008032c08d6c3f3424c13eb2ecc4bebacd3140', 'hex')::bytea AS expected_db_qkv,
        decode('bf4dcebe912dbf3f6d9ac83f97e99ebf341a053feaaff6bfad6c01c0a00dcd3fdd17f33ead17fdbfc4f6ffbf11b6d03fb11084bb55ff1ebe6450d5bde372f33d', 'hex')::bytea AS expected_dw_o,
        decode('ecb6e93e4194d8bfb441e3bfce06b43f', 'hex')::bytea AS expected_db_o
)
SELECT 'attention backward' AS label,
       encode(dx, 'hex') AS dx_hex,
       encode(expected_dx, 'hex') AS expected_dx_hex,
       encode(dw_qkv, 'hex') AS dw_qkv_hex,
       encode(expected_dw_qkv, 'hex') AS expected_dw_qkv_hex,
       encode(db_qkv, 'hex') AS db_qkv_hex,
       encode(expected_db_qkv, 'hex') AS expected_db_qkv_hex,
       encode(dw_o, 'hex') AS dw_o_hex,
       encode(expected_dw_o, 'hex') AS expected_dw_o_hex,
       encode(db_o, 'hex') AS db_o_hex,
       encode(expected_db_o, 'hex') AS expected_db_o_hex,
       dx = expected_dx AS dx_match,
       dw_qkv = expected_dw_qkv AS dw_qkv_match,
       db_qkv = expected_db_qkv AS db_qkv_match,
       dw_o = expected_dw_o AS dw_o_match,
       db_o = expected_db_o AS db_o_match
FROM fixture,
LATERAL (
    SELECT (pg_llm_attention_backward(x, w_qkv, b_qkv, w_o, b_o, dy, 2, 2, 4)).*
) AS run(dx, dw_qkv, db_qkv, dw_o, db_o);

WITH fixture AS (
    SELECT
        decode('29a5f63f535fbe3fe4aafebed811e13ee21442bf50068a3f44014d3f931ed73f4533b63e86c62fbf88471c3fb0d8aa3fed2e6dbe000c2b3d7ace80beb01f5c3f82949ebe919acabe', 'hex')::bytea AS x,
        decode('09ba64be80d2db3f4844a33e8d5ad9be74879c3eb24b46bf3f000f3d9d67a43ea53961bf72e419bf6317a3bfb6db07401d099ebfd9cff9befd82b5bfd771653ffbca9f3d679b063f0ddaf9bec87e983fce6250bf066a3cbfcd4d56bfc1216cbf5f0082bd19f52c3ffb4ec8bda81bec3fec9e97bf2518b13fbde999bf2f33353ff4f40d402dee053fac7bb13e41114abec5fc86bf5c95a33f66d3143ebb986c3e48ec673d7d43da3e8833133f084824bfa2350dc0a13440bf051934400c36b83e639babbfd8da15bfa043093fab4d063fe512923f3c88533d63653a3fa1ec35bfe85486bf1d991a3f2a74dcbfa9e853bf8cd7aa3f7692f73e9c4d4abe0758a23fafda483f81adea3c8a08243fb44f153f0e91883f817ae6bec1c22dbf6106133f6a3ccf3ecabf363e38a2873e2ef7a23f49d1abbaf4719bbe3e877cbf4084fc3d186219bfbd40f43ec0e6393fd4adba3d8e33c7be8925073fcffa833f7b6b34bf5ab7073e46a5433f47298c3f3006ae3ec749b6bf593ceebd280179bf7d5d753ff641cf3f95adb93f7cf9893ecd6c57be3935b8bf4579053f958fb23e40b4773fb76eeebeff69cd3f4aba1ec04bc8d5be', 'hex')::bytea AS w_qkv,
        decode('01fa45bea713273f7db5123d39f7f63dfa3f4ebf058f54be22946ebfc8a4cbbfa76791bff5c805bf54ce04bfd329c0bf9b9cf6bf6feb023ed3ee823f97480ebf384b343f6fba353f', 'hex')::bytea AS b_qkv,
        decode('291fe33fc4ea6bbf1e63763f408dacbe557196bf4253b73e1121f53e0b46ad3f01ac063fa12b0740575005bff4976ebfef9a3d3ee5ca883f853ca73f6e6feb3e638b50bff8b782bf9f66fdbece9d17bfb09912bfb825ab3d65c0cc3e579efe3f9618ecbee9d582bd1ef1aebf95dda83e475e903cf234a03d4acb453eb7c0d13e0ca7c9bfba0d10405228803f539fae3f', 'hex')::bytea AS w_o,
        decode('9ace05beb1c47a3ff691183f7175bcbfb722f3be06fdd43f', 'hex')::bytea AS b_o,
        decode('e868ffc03b58b8409a0915c03a8d884086ddb640806bcb40ea72eec090050b4164c3733ff35bad4039306b40fd7b3c40ae2d34c0aa149d40a85e23405a1d1f40c1489d3f328c6540', 'hex')::bytea AS expected
)
SELECT 'attention forward (T=3)' AS label,
       encode(actual, 'hex') AS actual_hex,
       encode(expected, 'hex') AS expected_hex,
       actual = expected AS matches
FROM fixture,
LATERAL (
    SELECT pg_llm_attention(x, w_qkv, b_qkv, w_o, b_o, 3, 3, 6) AS actual
) AS run;

WITH fixture AS (
    SELECT
        decode('29a5f63f535fbe3fe4aafebed811e13ee21442bf50068a3f44014d3f931ed73f4533b63e86c62fbf88471c3fb0d8aa3fed2e6dbe000c2b3d7ace80beb01f5c3f82949ebe919acabe', 'hex')::bytea AS x,
        decode('09ba64be80d2db3f4844a33e8d5ad9be74879c3eb24b46bf3f000f3d9d67a43ea53961bf72e419bf6317a3bfb6db07401d099ebfd9cff9befd82b5bfd771653ffbca9f3d679b063f0ddaf9bec87e983fce6250bf066a3cbfcd4d56bfc1216cbf5f0082bd19f52c3ffb4ec8bda81bec3fec9e97bf2518b13fbde999bf2f33353ff4f40d402dee053fac7bb13e41114abec5fc86bf5c95a33f66d3143ebb986c3e48ec673d7d43da3e8833133f084824bfa2350dc0a13440bf051934400c36b83e639babbfd8da15bfa043093fab4d063fe512923f3c88533d63653a3fa1ec35bfe85486bf1d991a3f2a74dcbfa9e853bf8cd7aa3f7692f73e9c4d4abe0758a23fafda483f81adea3c8a08243fb44f153f0e91883f817ae6bec1c22dbf6106133f6a3ccf3ecabf363e38a2873e2ef7a23f49d1abbaf4719bbe3e877cbf4084fc3d186219bfbd40f43ec0e6393fd4adba3d8e33c7be8925073fcffa833f7b6b34bf5ab7073e46a5433f47298c3f3006ae3ec749b6bf593ceebd280179bf7d5d753ff641cf3f95adb93f7cf9893ecd6c57be3935b8bf4579053f958fb23e40b4773fb76eeebeff69cd3f4aba1ec04bc8d5be', 'hex')::bytea AS w_qkv,
        decode('01fa45bea713273f7db5123d39f7f63dfa3f4ebf058f54be22946ebfc8a4cbbfa76791bff5c805bf54ce04bfd329c0bf9b9cf6bf6feb023ed3ee823f97480ebf384b343f6fba353f', 'hex')::bytea AS b_qkv,
        decode('291fe33fc4ea6bbf1e63763f408dacbe557196bf4253b73e1121f53e0b46ad3f01ac063fa12b0740575005bff4976ebfef9a3d3ee5ca883f853ca73f6e6feb3e638b50bff8b782bf9f66fdbece9d17bfb09912bfb825ab3d65c0cc3e579efe3f9618ecbee9d582bd1ef1aebf95dda83e475e903cf234a03d4acb453eb7c0d13e0ca7c9bfba0d10405228803f539fae3f', 'hex')::bytea AS w_o,
        decode('9ace05beb1c47a3ff691183f7175bcbfb722f3be06fdd43f', 'hex')::bytea AS b_o,
        decode('b97b2fbf6745103fd36f25c0b79ac4bc4240fabd183b3fbfa1cada3f59406d3d171e5d3fb9c816bff68c3a3fdb1b7b3f7045d43e430a943f85c1893ec00816bd0a26f6be42f2a13e', 'hex')::bytea AS dy,
        decode('eda3b740788d3dc08e87803f5f9506c0c643cc3eb2c83ac09a078fbf9a53ed3e0681423f63fe3e3f76c3bf3e40b1bebe3453febf0ba2dc3f96558abf606efc3ff72b0d3fcffebf3f', 'hex')::bytea AS expected_dx,
        decode('339fb93e96802a3e2a6c1a3eb174dc3d83e030bd4fd2803ce110ccbd2675b53f8d7c003f0cdcefbe106ef7bd572656bd84fde5bfdbed14c01f0c2bc0df5e7f3de7f603408822a7405e9e3b3f8898903e4701203e5063143d7cc313be37188e3dcbeab93bf82641be45c3893ec3e0f8bdb40ea5ba980177bb7cab6cbf9007aabf3d60babfb4e0013f7319c53fb8df80403b0a2d3e2e31be3d28a7e53d025ccf3dff8885bb966b55bb14b4883d7a2084bf6d20debd223d4d3e9cff983d00a1f43c07ab023f111b8e3ec16b5b3fd04c903e04f30ebfd5d7acbf2f19b1beb92c68be28f9a8be85eaa8bef97df7bcf370033d30d5aabd0785ab3f23cd24bba6173abe1f77b7bd11090cbde07590be9f360b3f0b7c36bf885a00bf0273023f32ef983fc51d913e904a133ec11b1f3e7e06053e8fc29ebc5c35383b30f7db3dd822d4bf15fb3cbeddada73e49bff63d72be453d13cf553f1c4f083fc1f0ae3f2f19e73ef2c55abf69bd03c0eed21a3f57b88e3e2131823e0a463b3e526591bd0844d13cd7ff283c01ee86bef8548a3e1b29e5bd9663723b5f91f7ba0afc53bf6d89c1bf385c93bfc58bf33e79578e3f7cd83a40', 'hex')::bytea AS expected_dw_qkv,
        decode('4ad0c43e681b353d76e74fbedf5fa8bebc743dbe1830de3d0000a6b3000098350000c0b300008032000070b4000076b332ec5cbeec41c33e8f9e31bfb7dd203e43f8843ff2542d40', 'hex')::bytea AS expected_db_qkv,
        decode('5165b0c0399c9fc044c4e340e72237409af2edbf9a77ddbfe61ee93f143adb3fa46c0fc095ad6cbf288c0d3f4e3f183facb3ad4058eb36409073dd3f250de4bf6839843fe64c464022048140a49ca140d54ac7c0ca5207c0ef344c3f02f7953fa7b94dc0967f7ec048e65440bd3bbb3fd25198be99989bbf1135643f1c898d3f0c646abf5e12cfbe745ba23d864ead3e', 'hex')::bytea AS expected_dw_o,
        decode('201eb83ffa96e33f39e0b9bf1b4e26bfa4c7003ee4d90c3f', 'hex')::bytea AS expected_db_o
)
SELECT 'attention backward (T=3)' AS label,
       encode(dx, 'hex') AS dx_hex,
       encode(expected_dx, 'hex') AS expected_dx_hex,
       encode(dw_qkv, 'hex') AS dw_qkv_hex,
       encode(expected_dw_qkv, 'hex') AS expected_dw_qkv_hex,
       encode(db_qkv, 'hex') AS db_qkv_hex,
       encode(expected_db_qkv, 'hex') AS expected_db_qkv_hex,
       encode(dw_o, 'hex') AS dw_o_hex,
       encode(expected_dw_o, 'hex') AS expected_dw_o_hex,
       encode(db_o, 'hex') AS db_o_hex,
       encode(expected_db_o, 'hex') AS expected_db_o_hex,
       dx = expected_dx AS dx_match,
       dw_qkv = expected_dw_qkv AS dw_qkv_match,
       db_qkv = expected_db_qkv AS db_qkv_match,
       dw_o = expected_dw_o AS dw_o_match,
       db_o = expected_db_o AS db_o_match
FROM fixture,
LATERAL (
    SELECT (pg_llm_attention_backward(x, w_qkv, b_qkv, w_o, b_o, dy, 3, 3, 6)).*
) AS run(dx, dw_qkv, db_qkv, dw_o, db_o);

WITH fixture AS (
    SELECT * FROM attention_fixture_ones
)
SELECT 'attention backward (dy=ones)' AS label,
       encode(dx, 'hex') AS dx_hex,
       encode(expected_dx, 'hex') AS expected_dx_hex,
       encode(dw_qkv, 'hex') AS dw_qkv_hex,
       encode(expected_dw_qkv, 'hex') AS expected_dw_qkv_hex,
       encode(db_qkv, 'hex') AS db_qkv_hex,
       encode(expected_db_qkv, 'hex') AS expected_db_qkv_hex,
       encode(dw_o, 'hex') AS dw_o_hex,
       encode(expected_dw_o, 'hex') AS expected_dw_o_hex,
       encode(db_o, 'hex') AS db_o_hex,
       encode(expected_db_o, 'hex') AS expected_db_o_hex,
       dx = expected_dx AS dx_match,
       dw_qkv = expected_dw_qkv AS dw_qkv_match,
       db_qkv = expected_db_qkv AS db_qkv_match,
       dw_o = expected_dw_o AS dw_o_match,
       db_o = expected_db_o AS db_o_match
FROM fixture,
LATERAL (
    SELECT (pg_llm_attention_backward(x, w_qkv, b_qkv, w_o, b_o, dy, 2, 2, 4)).*
) AS run(dx, dw_qkv, db_qkv, dw_o, db_o);

CREATE UNLOGGED TABLE IF NOT EXISTS llm_tape (
    id SERIAL PRIMARY KEY,
    name TEXT,
    inputs INT[],
    output INT,
    extra JSONB
);

CREATE UNLOGGED TABLE IF NOT EXISTS llm_autograd_mode (
    flag BOOL NOT NULL
);

CREATE UNLOGGED TABLE IF NOT EXISTS llm_tensor_rt (
    id SERIAL PRIMARY KEY,
    data BYTEA,
    grad BYTEA,
    shape INT[],
    requires_grad BOOL DEFAULT false
);

TRUNCATE llm_tape;
TRUNCATE llm_tensor_rt;
DELETE FROM llm_autograd_mode;
INSERT INTO llm_autograd_mode(flag) VALUES(true);

SELECT 'autograd attention forward' AS label,
       COUNT(*) AS calls
FROM (
    SELECT pg_llm_attention(x, w_qkv, b_qkv, w_o, b_o, 2, 2, 4)
    FROM attention_fixture_ones
) AS run;

DO $$
BEGIN
    PERFORM llm_backprop((SELECT MAX(id) FROM llm_tape));
END$$ LANGUAGE plpgsql;

WITH fixture AS (
    SELECT * FROM attention_fixture_ones
)
SELECT 'llm_backprop attention (dy=ones)' AS label,
       encode((SELECT grad FROM llm_tensor_rt WHERE data = fixture.x), 'hex') AS dx_hex,
       encode(fixture.expected_dx, 'hex') AS expected_dx_hex,
       encode((SELECT grad FROM llm_tensor_rt WHERE data = fixture.w_qkv), 'hex') AS dw_qkv_hex,
       encode(fixture.expected_dw_qkv, 'hex') AS expected_dw_qkv_hex,
       encode((SELECT grad FROM llm_tensor_rt WHERE data = fixture.b_qkv), 'hex') AS db_qkv_hex,
       encode(fixture.expected_db_qkv, 'hex') AS expected_db_qkv_hex,
       encode((SELECT grad FROM llm_tensor_rt WHERE data = fixture.w_o), 'hex') AS dw_o_hex,
       encode(fixture.expected_dw_o, 'hex') AS expected_dw_o_hex,
       encode((SELECT grad FROM llm_tensor_rt WHERE data = fixture.b_o), 'hex') AS db_o_hex,
       encode(fixture.expected_db_o, 'hex') AS expected_db_o_hex,
       (SELECT grad FROM llm_tensor_rt WHERE data = fixture.x) = fixture.expected_dx AS dx_match,
       (SELECT grad FROM llm_tensor_rt WHERE data = fixture.w_qkv) = fixture.expected_dw_qkv AS dw_qkv_match,
       (SELECT grad FROM llm_tensor_rt WHERE data = fixture.b_qkv) = fixture.expected_db_qkv AS db_qkv_match,
       (SELECT grad FROM llm_tensor_rt WHERE data = fixture.w_o) = fixture.expected_dw_o AS dw_o_match,
       (SELECT grad FROM llm_tensor_rt WHERE data = fixture.b_o) = fixture.expected_db_o AS db_o_match;

DELETE FROM llm_autograd_mode;

