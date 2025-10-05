#include "pg_llm.h"
#include <string.h>

#define PG_LLM_ATTENTION_CHUNK 64

static void
attention_compute(float *x, float *w_qkv, float *b_qkv,
                  float *w_o, float *b_o,
                  int n_head, int64_t T, int64_t D, float *Y)
{
    const int head_dim = D / n_head;
    const float scale = 1.0f / sqrtf((float)head_dim);
    float *qkv;
    float *Q;
    float *K;
    float *V;
    float *Q_chunk;
    float *score_chunk;
    float *context_chunk;
    float *K_head_T;
    float *V_head;
    float *proj;
    const int chunk_rows = Min(PG_LLM_ATTENTION_CHUNK, (int)T);
    const int pack_block = 32;

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
    K_head_T = (float *) palloc((Size)head_dim * T * sizeof(float));
    V_head = (float *) palloc((Size)T * head_dim * sizeof(float));

    for (int h = 0; h < n_head; ++h) {
        int off = h * head_dim;
        const float *K_head = K + off;
        const float *V_head_src = V + off;

        for (int t0 = 0; t0 < T; t0 += pack_block) {
            int t_block = Min(pack_block, (int)T - t0);

            for (int tt = 0; tt < t_block; ++tt) {
                const float *v_src = V_head_src + (Size)(t0 + tt) * D;
                float *v_dst = V_head + (Size)(t0 + tt) * head_dim;
                memcpy(v_dst, v_src, head_dim * sizeof(float));
            }

            for (int d0 = 0; d0 < head_dim; d0 += pack_block) {
                int d_block = Min(pack_block, head_dim - d0);
                for (int dd = 0; dd < d_block; ++dd) {
                    int d = d0 + dd;
                    float *k_dst = K_head_T + (Size)d * T + t0;
                    const float *k_src = K_head + (Size)t0 * D + d;
                    for (int tt = 0; tt < t_block; ++tt)
                        k_dst[tt] = k_src[(Size)tt * D];
                }
            }
        }

        for (int base = 0; base < T; base += chunk_rows) {
            int rows = Min(chunk_rows, (int)T - base);

            for (int r = 0; r < rows; ++r) {
                const float *src_row = Q + (Size)(base + r) * D + off;
                memcpy(Q_chunk + (Size)r * head_dim, src_row, head_dim * sizeof(float));
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

    }

    proj = (float *) palloc((Size)T * D * sizeof(float));
    pg_llm_fast_gemm(Y, w_o, proj, T, D, D);
    for (int t = 0; t < T; ++t) {
        float *dst = proj + (Size)t * D;
        for (int j = 0; j < D; ++j)
            dst[j] += b_o[j];
    }
    memcpy(Y, proj, (Size)T * D * sizeof(float));

    pfree(proj);
    pfree(V_head);
    pfree(K_head_T);
    pfree(context_chunk);
    pfree(score_chunk);
    pfree(Q_chunk);
    pfree(qkv);
}

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
PG_FUNCTION_INFO_V1(pg_llm_attention_backward);
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
    bool autograd = pg_llm_autograd_enabled();
    int input_ids[5];

    bytea *out = (bytea*) palloc(T*D*sizeof(float) + VARHDRSZ);
    SET_VARSIZE(out, T*D*sizeof(float) + VARHDRSZ);
    float *Y = as_float(out);

    if (autograd)
    {
        if (SPI_connect() != SPI_OK_CONNECT)
            ereport(ERROR, (errmsg("SPI_connect failed in pg_llm_attention")));

        PG_TRY();
        {
            int dims_x[2] = {(int)T, (int)D};
            int dims_wqkv[2] = {(int)D, (int)(3 * D)};
            int dims_bqkv[1] = {(int)(3 * D)};
            int dims_wo[2] = {(int)D, (int)D};
            int dims_bo[1] = {(int)D};
            input_ids[0] = pg_llm_autograd_track_tensor(x_b, 2, dims_x, true);
            input_ids[1] = pg_llm_autograd_track_tensor(w_qkvb, 2, dims_wqkv, true);
            input_ids[2] = pg_llm_autograd_track_tensor(b_qkvb, 1, dims_bqkv, true);
            input_ids[3] = pg_llm_autograd_track_tensor(w_ob, 2, dims_wo, true);
            input_ids[4] = pg_llm_autograd_track_tensor(b_ob, 1, dims_bo, true);

            attention_compute(x, w_qkv, b_qkv, w_o, b_o, n_head, T, D, Y);

            int dims_out[2] = {(int)T, (int)D};
            int output_id = pg_llm_autograd_track_tensor(out, 2, dims_out, true);
            char *extra = psprintf("{\"n_head\":%d,\"T\":%d,\"D\":%d}",
                                    n_head, (int)T, (int)D);
            pg_llm_autograd_record_tape("attention", input_ids, 5, output_id, extra);
        }
        PG_CATCH();
        {
            SPI_finish();
            PG_RE_THROW();
        }
        PG_END_TRY();

        SPI_finish();
    }
    else
    {
        attention_compute(x, w_qkv, b_qkv, w_o, b_o, n_head, T, D, Y);
    }

    PG_RETURN_BYTEA_P(out);
}

PG_FUNCTION_INFO_V1(pg_llm_attention_backward);
Datum
pg_llm_attention_backward(PG_FUNCTION_ARGS)
{
    bytea *x_b     = PG_GETARG_BYTEA_P(0);
    bytea *w_qkv_b = PG_GETARG_BYTEA_P(1);
    bytea *b_qkv_b = PG_GETARG_BYTEA_P(2);
    bytea *w_o_b   = PG_GETARG_BYTEA_P(3);
    bytea *b_o_b   = PG_GETARG_BYTEA_P(4);
    bytea *dy_b    = PG_GETARG_BYTEA_P(5);
    int n_head     = PG_GETARG_INT32(6);
    int64_t T      = PG_GETARG_INT32(7);
    int64_t D      = PG_GETARG_INT32(8);

    if (n_head <= 0 || D <= 0 || T <= 0)
        ereport(ERROR, (errmsg("pg_llm_attention_backward requires positive dimensions")));
    if (D % n_head != 0)
        ereport(ERROR, (errmsg("model dimension must be divisible by number of heads")));

    const int head_dim = (int) (D / n_head);
    const float scale = 1.0f / sqrtf((float) head_dim);

    float *x     = as_float(x_b);
    float *w_qkv = as_float(w_qkv_b);
    float *b_qkv = as_float(b_qkv_b);
    float *w_o   = as_float(w_o_b);
    float *dy    = as_float(dy_b);

    Size qkv_elems = (Size) T * 3 * D;
    Size td_elems = (Size) T * D;

    float *qkv = (float *) palloc(qkv_elems * sizeof(float));
    float *context = (float *) palloc(td_elems * sizeof(float));
    memset(context, 0, td_elems * sizeof(float));

    pg_llm_fast_gemm(x, w_qkv, qkv, (int) T, (int) D, (int) (3 * D));
    for (int64_t t = 0; t < T; ++t) {
        float *row = qkv + (Size) t * 3 * D;
        for (int64_t j = 0; j < 3 * D; ++j)
            row[j] += b_qkv[j];
    }

    float *Q = qkv;
    float *K = qkv + (Size) T * D;
    float *V = qkv + (Size) 2 * T * D;

    float *attn = (float *) palloc0((Size) n_head * T * T * sizeof(float));

    for (int h = 0; h < n_head; ++h) {
        int off = h * head_dim;
        float *attn_head = attn + (Size) h * T * T;
        for (int64_t t = 0; t < T; ++t) {
            float *row_attn = attn_head + (Size) t * T;
            int valid = (int) (t + 1);
            for (int64_t j = 0; j < valid; ++j) {
                float dot = 0.0f;
                const float *q_vec = Q + (Size) t * D + off;
                const float *k_vec = K + (Size) j * D + off;
                for (int d = 0; d < head_dim; ++d)
                    dot += q_vec[d] * k_vec[d];
                row_attn[j] = dot * scale;
            }
            for (int64_t j = valid; j < T; ++j)
                row_attn[j] = 0.0f;

            float maxv = -INFINITY;
            for (int j = 0; j < valid; ++j)
                if (row_attn[j] > maxv)
                    maxv = row_attn[j];

            float sum = 0.0f;
            for (int j = 0; j < valid; ++j) {
                float ex = expf(row_attn[j] - maxv);
                row_attn[j] = ex;
                sum += ex;
            }
            if (sum <= 0.0f)
                sum = 1.0f;
            float inv_sum = 1.0f / sum;
            for (int j = 0; j < valid; ++j)
                row_attn[j] *= inv_sum;
        }

        for (int64_t t = 0; t < T; ++t) {
            float *dst = context + (Size) t * D + off;
            float *attn_row = attn_head + (Size) t * T;
            for (int d = 0; d < head_dim; ++d) {
                float sum = 0.0f;
                for (int64_t j = 0; j <= t; ++j) {
                    float weight = attn_row[j];
                    sum += weight * V[(Size) j * D + off + d];
                }
                dst[d] = sum;
            }
        }
    }

    bytea *dx_b = bytea_same_size(x_b);
    bytea *dw_qkv_b = bytea_same_size(w_qkv_b);
    bytea *db_qkv_b = bytea_same_size(b_qkv_b);
    bytea *dw_o_b_out = bytea_same_size(w_o_b);
    bytea *db_o_b_out = bytea_same_size(b_o_b);

    float *dx = as_float(dx_b);
    float *dw_qkv_out = as_float(dw_qkv_b);
    float *db_qkv_out = as_float(db_qkv_b);
    float *dw_o_out = as_float(dw_o_b_out);
    float *db_o_out = as_float(db_o_b_out);

    memset(dx, 0, td_elems * sizeof(float));
    memset(dw_qkv_out, 0, ((Size) D * 3 * D) * sizeof(float));
    memset(db_qkv_out, 0, ((Size) 3 * D) * sizeof(float));
    memset(dw_o_out, 0, ((Size) D * D) * sizeof(float));
    memset(db_o_out, 0, ((Size) D) * sizeof(float));

    float *d_context = (float *) palloc0(td_elems * sizeof(float));

    for (int64_t t = 0; t < T; ++t) {
        const float *dy_row = dy + (Size) t * D;
        float *ctx_row = context + (Size) t * D;
        float *dctx_row = d_context + (Size) t * D;

        for (int64_t j = 0; j < D; ++j)
            db_o_out[j] += dy_row[j];

        for (int64_t i = 0; i < D; ++i)
            for (int64_t j = 0; j < D; ++j)
                dw_o_out[(Size) i * D + j] += ctx_row[i] * dy_row[j];

        for (int64_t i = 0; i < D; ++i) {
            float sum = 0.0f;
            for (int64_t j = 0; j < D; ++j)
                sum += dy_row[j] * w_o[(Size) i * D + j];
            dctx_row[i] = sum;
        }
    }

    float *dQ = (float *) palloc0(td_elems * sizeof(float));
    float *dK = (float *) palloc0(td_elems * sizeof(float));
    float *dV = (float *) palloc0(td_elems * sizeof(float));

    float *dqkv = (float *) palloc0(qkv_elems * sizeof(float));

    for (int h = 0; h < n_head; ++h) {
        int off = h * head_dim;
        float *attn_head = attn + (Size) h * T * T;
        float *dP = (float *) palloc0((Size) T * T * sizeof(float));
        float *dS = (float *) palloc0((Size) T * T * sizeof(float));

        for (int64_t t = 0; t < T; ++t) {
            const float *dctx_row = d_context + (Size) t * D + off;
            const float *attn_row = attn_head + (Size) t * T;
            for (int64_t j = 0; j <= t; ++j) {
                float dot = 0.0f;
                const float *v_vec = V + (Size) j * D + off;
                for (int d = 0; d < head_dim; ++d)
                    dot += dctx_row[d] * v_vec[d];
                dP[(Size) t * T + j] = dot;
                for (int d = 0; d < head_dim; ++d)
                    dV[(Size) j * D + off + d] += attn_row[j] * dctx_row[d];
            }
        }

        for (int64_t t = 0; t < T; ++t) {
            int valid = (int) (t + 1);
            const float *attn_row = attn_head + (Size) t * T;
            float sum = 0.0f;
            for (int j = 0; j < valid; ++j)
                sum += attn_row[j] * dP[(Size) t * T + j];
            for (int j = 0; j < valid; ++j) {
                float grad = attn_row[j] * (dP[(Size) t * T + j] - sum);
                dS[(Size) t * T + j] = grad;
            }
        }

        for (int64_t t = 0; t < T; ++t) {
            for (int64_t j = 0; j <= t; ++j) {
                float grad = dS[(Size) t * T + j] * scale;
                if (grad == 0.0f)
                    continue;
                const float *q_vec = Q + (Size) t * D + off;
                const float *k_vec = K + (Size) j * D + off;
                for (int d = 0; d < head_dim; ++d) {
                    dQ[(Size) t * D + off + d] += grad * k_vec[d];
                    dK[(Size) j * D + off + d] += grad * q_vec[d];
                }
            }
        }

        pfree(dP);
        pfree(dS);
    }

    for (int64_t t = 0; t < T; ++t) {
        float *dq_row = dqkv + (Size) t * 3 * D;
        for (int h = 0; h < n_head; ++h) {
            int off = h * head_dim;
            memcpy(dq_row + off,
                   dQ + (Size) t * D + off,
                   (Size) head_dim * sizeof(float));
            memcpy(dq_row + D + off,
                   dK + (Size) t * D + off,
                   (Size) head_dim * sizeof(float));
            memcpy(dq_row + 2 * D + off,
                   dV + (Size) t * D + off,
                   (Size) head_dim * sizeof(float));
        }
    }

    for (int64_t t = 0; t < T; ++t) {
        for (int64_t j = 0; j < 3 * D; ++j)
            db_qkv_out[j] += dqkv[(Size) t * 3 * D + j];
    }

    for (int64_t i = 0; i < D; ++i) {
        for (int64_t j = 0; j < 3 * D; ++j) {
            float sum = 0.0f;
            for (int64_t t = 0; t < T; ++t)
                sum += x[(Size) t * D + i] * dqkv[(Size) t * 3 * D + j];
            dw_qkv_out[(Size) i * 3 * D + j] = sum;
        }
    }

    for (int64_t t = 0; t < T; ++t) {
        for (int64_t i = 0; i < D; ++i) {
            float sum = 0.0f;
            for (int64_t j = 0; j < 3 * D; ++j)
                sum += dqkv[(Size) t * 3 * D + j] * w_qkv[(Size) i * 3 * D + j];
            dx[(Size) t * D + i] = sum;
        }
    }

    TupleDesc tupdesc;
    Datum values[5];
    bool nulls[5] = {false, false, false, false, false};

    if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
        ereport(ERROR, (errmsg("expected composite return type")));

    BlessTupleDesc(tupdesc);

    values[0] = PointerGetDatum(dx_b);
    values[1] = PointerGetDatum(dw_qkv_b);
    values[2] = PointerGetDatum(db_qkv_b);
    values[3] = PointerGetDatum(dw_o_b_out);
    values[4] = PointerGetDatum(db_o_b_out);

    HeapTuple rettuple = heap_form_tuple(tupdesc, values, nulls);

    pfree(dqkv);
    pfree(dV);
    pfree(dK);
    pfree(dQ);
    pfree(d_context);
    pfree(attn);
    pfree(context);
    pfree(qkv);

    PG_RETURN_DATUM(HeapTupleGetDatum(rettuple));
}
