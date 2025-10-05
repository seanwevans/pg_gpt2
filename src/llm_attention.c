#include "pg_llm.h"

#define PG_LLM_ATTENTION_CHUNK 64

/*
 * Multi-head self-attention
 * args:
 *   x:     BYTEA (T×D)
 *   w_qkv: BYTEA (D×3D)
 *   b_qkv: BYTEA (3D)
 *   w_o:   BYTEA (D×D)
 *   b_o:   BYTEA (D)
 *   n_head: INT
 * returns: BYTEA (T×D)
 */
PG_FUNCTION_INFO_V1(pg_llm_attention);
Datum pg_llm_attention(PG_FUNCTION_ARGS)
{
    bytea *x_b    = PG_GETARG_BYTEA_P(0);
    bytea *w_qkvb = PG_GETARG_BYTEA_P(1);
    bytea *b_qkvb = PG_GETARG_BYTEA_P(2);
    bytea *w_ob   = PG_GETARG_BYTEA_P(3);
    bytea *b_ob   = PG_GETARG_BYTEA_P(4);
    int n_head    = PG_GETARG_INT32(5);

    const int64_t T = PG_GETARG_INT32(6);  /* sequence length */
    const int64_t D = PG_GETARG_INT32(7);  /* model dim */

    float *x    = as_float(x_b);
    float *w_qkv= as_float(w_qkvb);
    float *w_o  = as_float(w_ob);
    float *b_qkv= as_float(b_qkvb);
    float *b_o  = as_float(b_ob);

    /* allocate temporary buffers */
    const int head_dim = D / n_head;
    const float scale = 1.0f / sqrtf((float)head_dim);
    bytea *out;
    float *Y;
    float *qkv;
    float *Q;
    float *K;
    float *V;
    float *Q_chunk;
    float *score_chunk;
    float *context_chunk;
    float *proj;
    const int chunk_rows = Min(PG_LLM_ATTENTION_CHUNK, (int)T);

    out = (bytea*) palloc(T*D*sizeof(float) + VARHDRSZ);
    SET_VARSIZE(out, T*D*sizeof(float) + VARHDRSZ);
    Y = as_float(out);

    /* 1. Compute Q,K,V using the optimized GEMM */
    qkv = (float *) palloc((Size)T * 3 * D * sizeof(float));
    pg_llm_fast_gemm(x, w_qkv, qkv, T, D, 3 * D);
    for (int t = 0; t < T; ++t) {
        float *row = qkv + (Size)t * 3 * D;
        for (int j = 0; j < 3 * D; ++j)
            row[j] += b_qkv[j];
    }
    Q = qkv;
    K = qkv + (Size)T * D;
    V = qkv + (Size)2 * T * D;

    Q_chunk = (float *) palloc((Size)chunk_rows * head_dim * sizeof(float));
    score_chunk = (float *) palloc((Size)chunk_rows * T * sizeof(float));
    context_chunk = (float *) palloc((Size)chunk_rows * head_dim * sizeof(float));

    /* 2. Iterate heads with chunked GEMMs to limit peak memory */
    for (int h = 0; h < n_head; ++h) {
        int off = h * head_dim;
        float *K_head_T = (float *) palloc((Size)head_dim * T * sizeof(float));
        float *V_head = (float *) palloc((Size)T * head_dim * sizeof(float));

        for (int t = 0; t < T; ++t) {
            const float *k_src = K + (Size)t * D + off;
            const float *v_src = V + (Size)t * D + off;
            float *v_dst = V_head + (Size)t * head_dim;
            memcpy(v_dst, v_src, head_dim * sizeof(float));
            for (int d = 0; d < head_dim; ++d)
                K_head_T[(Size)d * T + t] = k_src[d];
        }

        for (int base = 0; base < T; base += chunk_rows) {
            int rows = Min(chunk_rows, (int)T - base);

            for (int r = 0; r < rows; ++r) {
                const float *src = Q + (Size)(base + r) * D + off;
                memcpy(Q_chunk + (Size)r * head_dim, src, head_dim * sizeof(float));
            }

            pg_llm_fast_gemm(Q_chunk, K_head_T, score_chunk, rows, head_dim, T);

            for (int r = 0; r < rows; ++r) {
                int global_row = base + r;
                float *row = score_chunk + (Size)r * T;
                int valid = global_row + 1;
                float maxv = -INFINITY;
                for (int j = 0; j < T; ++j) {
                    if (j < valid) {
                        row[j] *= scale;
                        if (row[j] > maxv)
                            maxv = row[j];
                    } else {
                        row[j] = 0.0f;
                    }
                }
                float sum = 0.0f;
                for (int j = 0; j < valid; ++j) {
                    row[j] = expf(row[j] - maxv);
                    sum += row[j];
                }
                if (sum <= 0.0f)
                    sum = 1.0f;
                float inv_sum = 1.0f / sum;
                for (int j = 0; j < valid; ++j)
                    row[j] *= inv_sum;
                for (int j = valid; j < T; ++j)
                    row[j] = 0.0f;
            }

            pg_llm_fast_gemm(score_chunk, V_head, context_chunk, rows, T, head_dim);

            for (int r = 0; r < rows; ++r) {
                float *dst = Y + (Size)(base + r) * D + off;
                memcpy(dst, context_chunk + (Size)r * head_dim,
                       head_dim * sizeof(float));
            }
        }

        pfree(K_head_T);
        pfree(V_head);
    }

    /* 3. project output: Y = Y @ W_o */
    proj = (float *) palloc((Size)T * D * sizeof(float));
    pg_llm_fast_gemm(Y, w_o, proj, T, D, D);
    for (int t = 0; t < T; ++t) {
        float *dst = proj + (Size)t * D;
        for (int j = 0; j < D; ++j)
            dst[j] += b_o[j];
    }
    memcpy(Y, proj, (Size)T * D * sizeof(float));

    pfree(proj);
    pfree(context_chunk);
    pfree(score_chunk);
    pfree(Q_chunk);
    pfree(qkv);
    PG_RETURN_BYTEA_P(out);
}
