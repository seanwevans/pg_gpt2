#include "pg_llm.h"
#include <limits.h>
#include <string.h>

#define PG_LLM_ATTENTION_CHUNK 64

static Size
checked_mul_size(Size a, Size b, const char *context)
{
    if (a != 0 && b > SIZE_MAX / a)
        ereport(ERROR,
                (errmsg("%s size overflow", context)));
    return a * b;
}

static Size
bytea_num_floats(bytea *b, const char *fn_name)
{
    Size size = nbytes(b);

    if (size % sizeof(float) != 0)
        ereport(ERROR,
                (errmsg("%s expected a float32-aligned bytea (got %zu bytes)",
                        fn_name, size)));

    return size / sizeof(float);
}

static void
validate_attention_dims(const char *fn_name, int n_head, int64_t T, int64_t D)
{
    if (n_head <= 0)
        ereport(ERROR,
                (errmsg("%s requires a positive number of heads", fn_name)));
    if (T <= 0 || D <= 0)
        ereport(ERROR,
                (errmsg("%s requires positive sequence length and model dimension",
                        fn_name)));
    if (T > INT_MAX || D > INT_MAX)
        ereport(ERROR,
                (errmsg("%s dimensions exceed implementation limits (T=%lld, D=%lld)",
                        fn_name, (long long) T, (long long) D)));
    if (D % n_head != 0)
        ereport(ERROR,
                (errmsg("%s requires the model dimension to be divisible by number of heads",
                        fn_name)));
}

static void
attention_compute(float *x, float *w_qkv, float *b_qkv,
                  float *w_o, float *b_o,
                  int n_head, int64_t T, int64_t D, float *Y)
{
    const char *fn_name = "pg_llm_attention";
    const char *alloc_ctx = "pg_llm_attention workspace";

    validate_attention_dims(fn_name, n_head, T, D);

    const int head_dim = (int) (D / n_head);
    const float scale = 1.0f / sqrtf((float)head_dim);
    float *qkv = NULL;
    float *Q;
    float *K;
    float *V;
    float *Q_chunk = NULL;
    float *score_chunk = NULL;
    float *context_chunk = NULL;
    float *K_head_T = NULL;
    float *V_head = NULL;
    float *proj = NULL;
    const int chunk_rows = Min(PG_LLM_ATTENTION_CHUNK, (int)T);
    const int pack_block = 32;
    Size T_sz = (Size) T;
    Size D_sz = (Size) D;
    Size head_dim_sz = (Size) head_dim;
    Size chunk_rows_sz = (Size) chunk_rows;
    Size qkv_elems = checked_mul_size(checked_mul_size(T_sz, (Size) 3, alloc_ctx),
                                      D_sz,
                                      alloc_ctx);
    Size qkv_bytes = checked_mul_size(qkv_elems, sizeof(float), alloc_ctx);
    Size chunk_elems = checked_mul_size(chunk_rows_sz, head_dim_sz, alloc_ctx);
    Size chunk_bytes = checked_mul_size(chunk_elems, sizeof(float), alloc_ctx);
    Size score_elems = checked_mul_size(chunk_rows_sz, T_sz, alloc_ctx);
    Size score_bytes = checked_mul_size(score_elems, sizeof(float), alloc_ctx);
    Size head_T_elems = checked_mul_size(head_dim_sz, T_sz, alloc_ctx);
    Size head_T_bytes = checked_mul_size(head_T_elems, sizeof(float), alloc_ctx);
    Size proj_elems = checked_mul_size(T_sz, D_sz, alloc_ctx);
    Size proj_bytes = checked_mul_size(proj_elems, sizeof(float), alloc_ctx);

    PG_TRY();
    {
        qkv = (float *) palloc(qkv_bytes);
        pg_llm_fast_gemm(x, w_qkv, qkv, T, D, 3 * D);
        for (int t = 0; t < T; ++t) {
            float *row = qkv + (Size)t * 3 * D;
            for (int j = 0; j < 3 * D; ++j)
                row[j] += b_qkv[j];
        }
        Q = qkv;
        K = qkv + (Size)T * D;
        V = qkv + (Size)2 * T * D;

        Q_chunk = (float *) palloc(chunk_bytes);
        score_chunk = (float *) palloc(score_bytes);
        context_chunk = (float *) palloc(chunk_bytes);
        K_head_T = (float *) palloc(head_T_bytes);
        V_head = (float *) palloc(head_T_bytes);

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

        proj = (float *) palloc(proj_bytes);
        pg_llm_fast_gemm(Y, w_o, proj, T, D, D);
        for (int t = 0; t < T; ++t) {
            float *dst = proj + (Size)t * D;
            for (int j = 0; j < D; ++j)
                dst[j] += b_o[j];
        }
        memcpy(Y, proj, (Size)T * D * sizeof(float));

        pfree(proj);
        proj = NULL;
        pfree(V_head);
        V_head = NULL;
        pfree(K_head_T);
        K_head_T = NULL;
        pfree(context_chunk);
        context_chunk = NULL;
        pfree(score_chunk);
        score_chunk = NULL;
        pfree(Q_chunk);
        Q_chunk = NULL;
        pfree(qkv);
        qkv = NULL;
    }
    PG_CATCH();
    {
        if (proj)
            pfree(proj);
        if (V_head)
            pfree(V_head);
        if (K_head_T)
            pfree(K_head_T);
        if (context_chunk)
            pfree(context_chunk);
        if (score_chunk)
            pfree(score_chunk);
        if (Q_chunk)
            pfree(Q_chunk);
        if (qkv)
            pfree(qkv);
        PG_RE_THROW();
    }
    PG_END_TRY();
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

    const char *fn_name = "pg_llm_attention";
    Size T_sz;
    Size D_sz;
    Size td_elems;
    Size td_bytes;
    Size qkv_param_elems;
    Size b_qkv_elems;
    Size proj_param_elems;
    Size bias_proj_elems;

    validate_attention_dims(fn_name, n_head, T, D);

    T_sz = (Size) T;
    D_sz = (Size) D;
    td_elems = checked_mul_size(T_sz, D_sz, "pg_llm_attention activations elements");
    td_bytes = checked_mul_size(td_elems, sizeof(float), "pg_llm_attention activations bytes");
    qkv_param_elems = checked_mul_size(checked_mul_size(D_sz, (Size) 3, "pg_llm_attention qkv param elements"),
                                       D_sz,
                                       "pg_llm_attention qkv param elements");
    b_qkv_elems = checked_mul_size((Size) 3, D_sz, "pg_llm_attention qkv bias elements");
    proj_param_elems = checked_mul_size(D_sz, D_sz, "pg_llm_attention projection param elements");
    bias_proj_elems = D_sz;

    Size x_elems = bytea_num_floats(x_b, fn_name);
    Size w_qkv_elems = bytea_num_floats(w_qkvb, fn_name);
    Size b_qkv_actual = bytea_num_floats(b_qkvb, fn_name);
    Size w_o_elems = bytea_num_floats(w_ob, fn_name);
    Size b_o_elems = bytea_num_floats(b_ob, fn_name);

    if (x_elems != td_elems)
        ereport(ERROR,
                (errmsg("pg_llm_attention expected x with %zu floats (got %zu)",
                        td_elems, x_elems)));
    if (w_qkv_elems != qkv_param_elems)
        ereport(ERROR,
                (errmsg("pg_llm_attention expected w_qkv with %zu floats (got %zu)",
                        qkv_param_elems, w_qkv_elems)));
    if (b_qkv_actual != b_qkv_elems)
        ereport(ERROR,
                (errmsg("pg_llm_attention expected b_qkv with %zu floats (got %zu)",
                        b_qkv_elems, b_qkv_actual)));
    if (w_o_elems != proj_param_elems)
        ereport(ERROR,
                (errmsg("pg_llm_attention expected w_o with %zu floats (got %zu)",
                        proj_param_elems, w_o_elems)));
    if (b_o_elems != bias_proj_elems)
        ereport(ERROR,
                (errmsg("pg_llm_attention expected b_o with %zu floats (got %zu)",
                        bias_proj_elems, b_o_elems)));

    float *x     = as_float(x_b);
    float *w_qkv = as_float(w_qkvb);
    float *w_o   = as_float(w_ob);
    float *b_qkv = as_float(b_qkvb);
    float *b_o   = as_float(b_ob);
    bool autograd = pg_llm_autograd_enabled();
    int input_ids[5];

    bytea *out = bytea_alloc(td_bytes);
    float *Y = as_float(out);

    bool spi_connected = false;

    PG_TRY();
    {
        if (autograd)
        {
            if (SPI_connect() != SPI_OK_CONNECT)
                ereport(ERROR, (errmsg("SPI_connect failed in pg_llm_attention")));
            spi_connected = true;

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
            pfree(extra);
        }
        else
        {
            attention_compute(x, w_qkv, b_qkv, w_o, b_o, n_head, T, D, Y);
        }
    }
    PG_CATCH();
    {
        if (spi_connected)
            SPI_finish();
        if (out)
            pfree(out);
        PG_RE_THROW();
    }
    PG_END_TRY();

    if (spi_connected)
        SPI_finish();

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

    const char *fn_name = "pg_llm_attention_backward";
    Size T_sz;
    Size D_sz;
    Size n_head_sz;
    Size td_elems;
    Size td_bytes;
    Size qkv_elems;
    Size qkv_bytes;
    Size param_qkv_elems;
    Size param_qkv_bytes;
    Size b_qkv_elems;
    Size b_qkv_bytes;
    Size proj_param_elems;
    Size proj_param_bytes;
    Size bias_proj_elems;
    Size bias_proj_bytes;
    Size attn_elems;
    Size attn_bytes;
    Size tt_elems;
    Size tt_bytes;

    validate_attention_dims(fn_name, n_head, T, D);

    T_sz = (Size) T;
    D_sz = (Size) D;
    n_head_sz = (Size) n_head;
    td_elems = checked_mul_size(T_sz, D_sz, "pg_llm_attention_backward activations elements");
    td_bytes = checked_mul_size(td_elems, sizeof(float), "pg_llm_attention_backward activations bytes");
    qkv_elems = checked_mul_size(checked_mul_size(T_sz, (Size) 3, "pg_llm_attention_backward qkv workspace elements"),
                                 D_sz,
                                 "pg_llm_attention_backward qkv workspace elements");
    qkv_bytes = checked_mul_size(qkv_elems, sizeof(float), "pg_llm_attention_backward qkv workspace bytes");
    param_qkv_elems = checked_mul_size(checked_mul_size(D_sz, (Size) 3, "pg_llm_attention_backward qkv param elements"),
                                       D_sz,
                                       "pg_llm_attention_backward qkv param elements");
    param_qkv_bytes = checked_mul_size(param_qkv_elems, sizeof(float), "pg_llm_attention_backward qkv param bytes");
    b_qkv_elems = checked_mul_size((Size) 3, D_sz, "pg_llm_attention_backward qkv bias elements");
    b_qkv_bytes = checked_mul_size(b_qkv_elems, sizeof(float), "pg_llm_attention_backward qkv bias bytes");
    proj_param_elems = checked_mul_size(D_sz, D_sz, "pg_llm_attention_backward projection param elements");
    proj_param_bytes = checked_mul_size(proj_param_elems, sizeof(float), "pg_llm_attention_backward projection param bytes");
    bias_proj_elems = D_sz;
    bias_proj_bytes = checked_mul_size(bias_proj_elems, sizeof(float), "pg_llm_attention_backward projection bias bytes");
    tt_elems = checked_mul_size(T_sz, T_sz, "pg_llm_attention_backward time-step square elements");
    tt_bytes = checked_mul_size(tt_elems, sizeof(float), "pg_llm_attention_backward time-step square bytes");
    attn_elems = checked_mul_size(n_head_sz, tt_elems, "pg_llm_attention_backward attention tensor elements");
    attn_bytes = checked_mul_size(attn_elems, sizeof(float), "pg_llm_attention_backward attention tensor bytes");

    Size x_elems = bytea_num_floats(x_b, fn_name);
    Size w_qkv_param = bytea_num_floats(w_qkv_b, fn_name);
    Size b_qkv_param = bytea_num_floats(b_qkv_b, fn_name);
    Size w_o_param = bytea_num_floats(w_o_b, fn_name);
    Size b_o_param = bytea_num_floats(b_o_b, fn_name);
    Size dy_elems = bytea_num_floats(dy_b, fn_name);

    if (x_elems != td_elems)
        ereport(ERROR,
                (errmsg("pg_llm_attention_backward expected x with %zu floats (got %zu)",
                        td_elems, x_elems)));
    if (dy_elems != td_elems)
        ereport(ERROR,
                (errmsg("pg_llm_attention_backward expected dy with %zu floats (got %zu)",
                        td_elems, dy_elems)));
    if (w_qkv_param != param_qkv_elems)
        ereport(ERROR,
                (errmsg("pg_llm_attention_backward expected w_qkv with %zu floats (got %zu)",
                        param_qkv_elems, w_qkv_param)));
    if (b_qkv_param != b_qkv_elems)
        ereport(ERROR,
                (errmsg("pg_llm_attention_backward expected b_qkv with %zu floats (got %zu)",
                        b_qkv_elems, b_qkv_param)));
    if (w_o_param != proj_param_elems)
        ereport(ERROR,
                (errmsg("pg_llm_attention_backward expected w_o with %zu floats (got %zu)",
                        proj_param_elems, w_o_param)));
    if (b_o_param != bias_proj_elems)
        ereport(ERROR,
                (errmsg("pg_llm_attention_backward expected b_o with %zu floats (got %zu)",
                        bias_proj_elems, b_o_param)));

    const int head_dim = (int) (D / n_head);
    const float scale = 1.0f / sqrtf((float) head_dim);

    float *x     = as_float(x_b);
    float *w_qkv = as_float(w_qkv_b);
    float *b_qkv = as_float(b_qkv_b);
    float *w_o   = as_float(w_o_b);
    float *dy    = as_float(dy_b);

    float *qkv = NULL;
    float *context = NULL;
    float *attn = NULL;
    bytea *dx_b = NULL;
    bytea *dw_qkv_b = NULL;
    bytea *db_qkv_b = NULL;
    bytea *dw_o_b_out = NULL;
    bytea *db_o_b_out = NULL;
    float *dx = NULL;
    float *dw_qkv_out = NULL;
    float *db_qkv_out = NULL;
    float *dw_o_out = NULL;
    float *db_o_out = NULL;
    float *d_context = NULL;
    float *dQ = NULL;
    float *dK = NULL;
    float *dV = NULL;
    float *dqkv = NULL;
    float *dP = NULL;
    float *dS = NULL;
    TupleDesc tupdesc;
    Datum values[5];
    bool nulls[5] = {false, false, false, false, false};
    HeapTuple rettuple = NULL;
    Datum result = (Datum) 0;

    PG_TRY();
    {
        qkv = (float *) palloc(qkv_bytes);
        context = (float *) palloc(td_bytes);
        memset(context, 0, td_bytes);

        pg_llm_fast_gemm(x, w_qkv, qkv, (int) T, (int) D, (int) (3 * D));
        for (int64_t t = 0; t < T; ++t) {
            float *row = qkv + (Size) t * 3 * D;
            for (int64_t j = 0; j < 3 * D; ++j)
                row[j] += b_qkv[j];
        }

        float *Q = qkv;
        float *K = qkv + (Size) T * D;
        float *V = qkv + (Size) 2 * T * D;

        attn = (float *) palloc0(attn_bytes);

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

        dx_b = bytea_same_size(x_b);
        dw_qkv_b = bytea_same_size(w_qkv_b);
        db_qkv_b = bytea_same_size(b_qkv_b);
        dw_o_b_out = bytea_same_size(w_o_b);
        db_o_b_out = bytea_same_size(b_o_b);

        dx = as_float(dx_b);
        dw_qkv_out = as_float(dw_qkv_b);
        db_qkv_out = as_float(db_qkv_b);
        dw_o_out = as_float(dw_o_b_out);
        db_o_out = as_float(db_o_b_out);

        memset(dx, 0, td_bytes);
        memset(dw_qkv_out, 0, param_qkv_bytes);
        memset(db_qkv_out, 0, b_qkv_bytes);
        memset(dw_o_out, 0, proj_param_bytes);
        memset(db_o_out, 0, bias_proj_bytes);

        d_context = (float *) palloc0(td_bytes);

        for (int64_t t = 0; t < T; ++t) {
            const float *dy_row = dy + (Size) t * D;
            float *ctx_row = context + (Size) t * D;
            float *dctx_row = d_context + (Size) t * D;

            for (int64_t j = 0; j < D; ++j)
                db_o_out[j] += dy_row[j];

            for (int64_t i = 0; i < D; ++i)
                for (int64_t j = 0; j < D; ++j)
                    dw_o_out[(Size) i * D + j] += ctx_row[i] * dy_row[j];

            for (int64_t i = 0; i < D; ++i)
                for (int64_t j = 0; j < D; ++j)
                    dctx_row[i] += dy_row[j] * w_o[(Size) i * D + j];
        }

        dQ = (float *) palloc0(td_bytes);
        dK = (float *) palloc0(td_bytes);
        dV = (float *) palloc0(td_bytes);

        dqkv = (float *) palloc0(qkv_bytes);

        for (int h = 0; h < n_head; ++h) {
            int off = h * head_dim;
            float *attn_head = attn + (Size) h * T * T;
            dP = (float *) palloc0(tt_bytes);
            dS = (float *) palloc0(tt_bytes);

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
            dP = NULL;
            pfree(dS);
            dS = NULL;
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

        if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
            ereport(ERROR, (errmsg("expected composite return type")));

        BlessTupleDesc(tupdesc);

        values[0] = PointerGetDatum(dx_b);
        values[1] = PointerGetDatum(dw_qkv_b);
        values[2] = PointerGetDatum(db_qkv_b);
        values[3] = PointerGetDatum(dw_o_b_out);
        values[4] = PointerGetDatum(db_o_b_out);

        rettuple = heap_form_tuple(tupdesc, values, nulls);

        pfree(dqkv);
        dqkv = NULL;
        pfree(dV);
        dV = NULL;
        pfree(dK);
        dK = NULL;
        pfree(dQ);
        dQ = NULL;
        pfree(d_context);
        d_context = NULL;
        pfree(attn);
        attn = NULL;
        pfree(context);
        context = NULL;
        pfree(qkv);
        qkv = NULL;

        result = HeapTupleGetDatum(rettuple);
    }
    PG_CATCH();
    {
        if (dP)
            pfree(dP);
        if (dS)
            pfree(dS);
        if (dqkv)
            pfree(dqkv);
        if (dV)
            pfree(dV);
        if (dK)
            pfree(dK);
        if (dQ)
            pfree(dQ);
        if (d_context)
            pfree(d_context);
        if (attn)
            pfree(attn);
        if (context)
            pfree(context);
        if (qkv)
            pfree(qkv);
        PG_RE_THROW();
    }
    PG_END_TRY();

    PG_RETURN_DATUM(result);
}


