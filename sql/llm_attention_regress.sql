-- Regression harness for pg_llm_attention.
--
-- Compares the C implementation against a precomputed GPT-2 style
-- attention result for a tiny fixture (T=2, D=4, n_head=2).  The
-- fixtures were generated via a float32 NumPy reference implementation
-- mirroring GPT-2 attention math.
WITH fixture AS (
    SELECT
        decode('5151fe3e28950dbeebce253fa4f2c23fe7c56fbe99c16fbea523ca3f9a76443f', 'hex')::bytea AS x,
        decode('f25ef0be37e50a3f1545edbe2174eebef5c4773e5ee6f4bf1ccadcbf13f20fbf73a481bf07e5a03e447468bf5ec6b4bf619abb3fe73167be3a4c8a3d265eb6bfab5c0bbf622be33dc25393bf7e5bc03e75c319bfe25895be72091abf7417ed3f77235dbc126387bf4e92523f9b449cbf57e0553e78d6fabf0002aabffe95493e250c3d3f2b7b2f3e02d9ecbd432a9abe3540bdbfb64738bfd8d8ebbec84f873fbdeeaf3e4dabe1bf56eea53e8129c5bec34a2dbfd1961c3fcbf7833f60686e3f', 'hex')::bytea AS w_qkv,
        decode('0ad7233c0ad7a3bc8fc2f53c0ad723bdcdcc4c3d8fc275bd295c8f3d0ad7a3bdec51b83dcdccccbdae47e13d8fc2f5bd', 'hex')::bytea AS b_qkv,
        decode('f6d656bf16519ebe5c9ba93e53bd793f5356f5be631d3ebe629c8dbf4c1d99bfb101503f4699ad3f0b7a93bdc473803f5b28b93e912625bfd808b93e62dec43f', 'hex')::bytea AS w_o,
        decode('0ad7a33b8fc275bccdcccc3c295c0fbd', 'hex')::bytea AS b_o,
        decode('0a0da13fa9bed03fe230c8bfa7e84abfafd9603f9a68bb3fdae9a7bf3ab6bfbe', 'hex')::bytea AS expected
)
SELECT
    actual_hex,
    encode(expected, 'hex') AS expected_hex,
    actual_hex = encode(expected, 'hex') AS matches
FROM fixture,
LATERAL (
    SELECT encode(pg_llm_attention(x, w_qkv, b_qkv, w_o, b_o, 2, 2, 4), 'hex') AS actual_hex
) ref;
