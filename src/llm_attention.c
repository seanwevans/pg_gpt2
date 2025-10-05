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

Datum
pg_llm_attention_backward(PG_FUNCTION_ARGS)
{
    bytea *x_b = PG_GETARG_BYTEA_P(0);
    bytea *w_qkv_b = PG_GETARG_BYTEA_P(1);
    bytea *b_qkv_b = PG_GETARG_BYTEA_P(2);
    bytea *w_o_b = PG_GETARG_BYTEA_P(3);
    bytea *b_o_b = PG_GETARG_BYTEA_P(4);
    bytea *grad_out_b = PG_GETARG_BYTEA_P(5);
    int n_head = PG_GETARG_INT32(6);
    int T = PG_GETARG_INT32(7);
    int D = PG_GETARG_INT32(8);

    const Size td = (Size) T * (Size) D;
    const Size threeD = (Size) 3 * (Size) D;
    const int head_dim = (n_head > 0) ? D / n_head : 0;
    const float scale = head_dim > 0 ? 1.0f / sqrtf((float) head_dim) : 0.0f;

    float *x;
    float *w_qkv;
    float *b_qkv;
    float *w_o;
    float *grad_out;
    bytea *dx_b_out;
    bytea *dw_qkv_b_out;
    bytea *db_qkv_b_out;
    bytea *dw_o_b_out;
    bytea *db_o_b_out;
    float *dx;
    float *dw_qkv;
    float *db_qkv;
    float *dw_o_grad;
    float *db_o_grad;
    float *qkv;
    float *Q;
    float *K;
    float *V;
    float *context;
    float *dcontext;
    float *dQ;
    float *dK;
    float *dV;
    float *dqkv;
    TupleDesc tupdesc;
    Datum values[5];
    bool nulls[5] = {false, false, false, false, false};
    HeapTuple rettuple;

    if (T <= 0 || D <= 0)
        ereport(ERROR,
                (errmsg("pg_llm_attention_backward requires positive dimensions")));
    if (n_head <= 0 || D % n_head != 0)
        ereport(ERROR,
                (errmsg("pg_llm_attention_backward requires n_head dividing D")));
    if ((Size) head_dim * (Size) n_head != (Size) D)
        ereport(ERROR,
                (errmsg("pg_llm_attention_backward invalid head dimension")));

    if (nbytes(x_b) != td * sizeof(float))
        ereport(ERROR,
                (errmsg("pg_llm_attention_backward expected x to have %zu bytes", td * sizeof(float))));
    if (nbytes(w_qkv_b) != (Size) D * threeD * sizeof(float))
        ereport(ERROR,
                (errmsg("pg_llm_attention_backward expected w_qkv to have %zu bytes",
                        (Size) D * threeD * sizeof(float))));
    if (nbytes(b_qkv_b) != threeD * sizeof(float))
        ereport(ERROR,
                (errmsg("pg_llm_attention_backward expected b_qkv to have %zu bytes",
                        threeD * sizeof(float))));
    if (nbytes(w_o_b) != (Size) D * (Size) D * sizeof(float))
        ereport(ERROR,
                (errmsg("pg_llm_attention_backward expected w_o to have %zu bytes",
                        (Size) D * (Size) D * sizeof(float))));
    if (nbytes(b_o_b) != (Size) D * sizeof(float))
        ereport(ERROR,
                (errmsg("pg_llm_attention_backward expected b_o to have %zu bytes",
                        (Size) D * sizeof(float))));
    if (nbytes(grad_out_b) != td * sizeof(float))
        ereport(ERROR,
                (errmsg("pg_llm_attention_backward expected grad_output to have %zu bytes",
                        td * sizeof(float))));

    x = as_float(x_b);
    w_qkv = as_float(w_qkv_b);
    b_qkv = as_float(b_qkv_b);
    w_o = as_float(w_o_b);
    grad_out = as_float(grad_out_b);

    dx_b_out = bytea_same_size(x_b);
    dw_qkv_b_out = bytea_same_size(w_qkv_b);
    db_qkv_b_out = bytea_same_size(b_qkv_b);
    dw_o_b_out = bytea_same_size(w_o_b);
    db_o_b_out = bytea_same_size(b_o_b);

    dx = as_float(dx_b_out);
    dw_qkv = as_float(dw_qkv_b_out);
    db_qkv = as_float(db_qkv_b_out);
    dw_o_grad = as_float(dw_o_b_out);
    db_o_grad = as_float(db_o_b_out);

    memset(dx, 0, td * sizeof(float));
    memset(dw_qkv, 0, (Size) D * threeD * sizeof(float));
    memset(db_qkv, 0, threeD * sizeof(float));
    memset(dw_o_grad, 0, (Size) D * (Size) D * sizeof(float));
    memset(db_o_grad, 0, (Size) D * sizeof(float));

    qkv = (float *) palloc((Size) T * threeD * sizeof(float));
    pg_llm_fast_gemm(x, w_qkv, qkv, T, D, 3 * D);
    for (int t = 0; t < T; ++t)
    {
        float *row = qkv + (Size) t * 3 * D;
        for (int j = 0; j < 3 * D; ++j)
            row[j] += b_qkv[j];
    }
    Q = qkv;
    K = qkv + (Size) T * D;
    V = qkv + (Size) 2 * T * D;

    context = (float *) palloc0(td * sizeof(float));
    {
        float *scores = (float *) palloc((Size) T * sizeof(float));
        float *attn = (float *) palloc((Size) T * sizeof(float));
        for (int h = 0; h < n_head; ++h)
        {
            int off = h * head_dim;
            for (int i = 0; i < T; ++i)
            {
                int valid = i + 1;
                const float *Q_row = Q + (Size) i * D + off;
                float *ctx_row = context + (Size) i * D + off;
                float maxv = -INFINITY;
                for (int j = 0; j < valid; ++j)
                {
                    const float *K_row = K + (Size) j * D + off;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d)
                        dot += Q_row[d] * K_row[d];
                    scores[j] = dot * scale;
                    if (scores[j] > maxv)
                        maxv = scores[j];
                }
                float sum = 0.0f;
                for (int j = 0; j < valid; ++j)
                {
                    float val = expf(scores[j] - maxv);
                    attn[j] = val;
                    sum += val;
                }
                float inv_sum = 1.0f / (sum > 0.0f ? sum : 1.0f);
                memset(ctx_row, 0, head_dim * sizeof(float));
                for (int j = 0; j < valid; ++j)
                {
                    const float *V_row = V + (Size) j * D + off;
                    float weight = attn[j] * inv_sum;
                    for (int d = 0; d < head_dim; ++d)
                        ctx_row[d] += weight * V_row[d];
                }
            }
        }
        pfree(scores);
        pfree(attn);
    }

    dcontext = (float *) palloc0(td * sizeof(float));
    for (int col = 0; col < D; ++col)
    {
        float sum = 0.0f;
        for (int t = 0; t < T; ++t)
            sum += grad_out[(Size) t * D + col];
        db_o_grad[col] = sum;
    }
    for (int row = 0; row < D; ++row)
    {
        for (int col = 0; col < D; ++col)
        {
            float sum = 0.0f;
            for (int t = 0; t < T; ++t)
                sum += context[(Size) t * D + row] * grad_out[(Size) t * D + col];
            dw_o_grad[(Size) row * D + col] = sum;
        }
    }
    for (int t = 0; t < T; ++t)
    {
        for (int row = 0; row < D; ++row)
        {
            float sum = 0.0f;
            for (int col = 0; col < D; ++col)
                sum += grad_out[(Size) t * D + col] * w_o[(Size) row * D + col];
            dcontext[(Size) t * D + row] = sum;
        }
    }

    dQ = (float *) palloc0(td * sizeof(float));
    dK = (float *) palloc0(td * sizeof(float));
    dV = (float *) palloc0(td * sizeof(float));

    {
        float *scores = (float *) palloc((Size) T * sizeof(float));
        float *attn = (float *) palloc((Size) T * sizeof(float));
        float *dattn = (float *) palloc((Size) T * sizeof(float));
        float *dscores = (float *) palloc((Size) T * sizeof(float));
        for (int h = 0; h < n_head; ++h)
        {
            int off = h * head_dim;
            for (int i = 0; i < T; ++i)
            {
                int valid = i + 1;
                const float *Q_row = Q + (Size) i * D + off;
                const float *dctx_row = dcontext + (Size) i * D + off;
                float *dQ_row = dQ + (Size) i * D + off;
                float maxv = -INFINITY;
                for (int j = 0; j < valid; ++j)
                {
                    const float *K_row = K + (Size) j * D + off;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d)
                        dot += Q_row[d] * K_row[d];
                    scores[j] = dot * scale;
                    if (scores[j] > maxv)
                        maxv = scores[j];
                }
                float sum = 0.0f;
                for (int j = 0; j < valid; ++j)
                {
                    float val = expf(scores[j] - maxv);
                    attn[j] = val;
                    sum += val;
                }
                float inv_sum = 1.0f / (sum > 0.0f ? sum : 1.0f);
                for (int j = 0; j < valid; ++j)
                    attn[j] *= inv_sum;

                for (int j = 0; j < valid; ++j)
                {
                    float *dV_row = dV + (Size) j * D + off;
                    for (int d = 0; d < head_dim; ++d)
                        dV_row[d] += attn[j] * dctx_row[d];
                }

                float dot = 0.0f;
                for (int j = 0; j < valid; ++j)
                {
                    const float *V_row = V + (Size) j * D + off;
                    float val = 0.0f;
                    for (int d = 0; d < head_dim; ++d)
                        val += dctx_row[d] * V_row[d];
                    dattn[j] = val;
                    dot += val * attn[j];
                }
                for (int j = 0; j < valid; ++j)
                    dscores[j] = (dattn[j] - dot) * attn[j];

                for (int j = 0; j < valid; ++j)
                {
                    const float *K_row = K + (Size) j * D + off;
                    float coeff = dscores[j] * scale;
                    float *dK_row = dK + (Size) j * D + off;
                    for (int d = 0; d < head_dim; ++d)
                    {
                        dQ_row[d] += coeff * K_row[d];
                        dK_row[d] += coeff * Q_row[d];
                    }
                }
            }
        }
        pfree(scores);
        pfree(attn);
        pfree(dattn);
        pfree(dscores);
    }

    dqkv = (float *) palloc0((Size) T * 3 * D * sizeof(float));
    for (int t = 0; t < T; ++t)
    {
        float *dst = dqkv + (Size) t * 3 * D;
        memcpy(dst, dQ + (Size) t * D, D * sizeof(float));
        memcpy(dst + D, dK + (Size) t * D, D * sizeof(float));
        memcpy(dst + 2 * D, dV + (Size) t * D, D * sizeof(float));
    }

    for (int t = 0; t < T; ++t)
    {
        float *dx_row = dx + (Size) t * D;
        const float *dqkv_row = dqkv + (Size) t * 3 * D;
        for (int di = 0; di < D; ++di)
        {
            float sum = 0.0f;
            for (int j = 0; j < 3 * D; ++j)
                sum += dqkv_row[j] * w_qkv[(Size) di * 3 * D + j];
            dx_row[di] = sum;
        }
    }

    for (int di = 0; di < D; ++di)
    {
        for (int j = 0; j < 3 * D; ++j)
        {
            float sum = 0.0f;
            for (int t = 0; t < T; ++t)
                sum += x[(Size) t * D + di] * dqkv[(Size) t * 3 * D + j];
            dw_qkv[(Size) di * 3 * D + j] = sum;
        }
    }
    for (int j = 0; j < 3 * D; ++j)
    {
        float sum = 0.0f;
        for (int t = 0; t < T; ++t)
            sum += dqkv[(Size) t * 3 * D + j];
        db_qkv[j] = sum;
    }

    if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
        ereport(ERROR,
                (errmsg("pg_llm_attention_backward expected composite return type")));
    BlessTupleDesc(tupdesc);

    values[0] = PointerGetDatum(dx_b_out);
    values[1] = PointerGetDatum(dw_qkv_b_out);
    values[2] = PointerGetDatum(db_qkv_b_out);
    values[3] = PointerGetDatum(dw_o_b_out);
    values[4] = PointerGetDatum(db_o_b_out);

    rettuple = heap_form_tuple(tupdesc, values, nulls);

    pfree(qkv);
    pfree(context);
    pfree(dcontext);
    pfree(dQ);
    pfree(dK);
    pfree(dV);
    pfree(dqkv);

    PG_RETURN_DATUM(HeapTupleGetDatum(rettuple));
}
