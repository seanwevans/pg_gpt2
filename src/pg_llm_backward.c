#include "pg_llm.h"
#include <string.h>

PG_FUNCTION_INFO_V1(pg_llm_gelu_backward);
PG_FUNCTION_INFO_V1(pg_llm_softmax_backward);
PG_FUNCTION_INFO_V1(pg_llm_layernorm_backward);
PG_FUNCTION_INFO_V1(pg_llm_dropout_backward);
PG_FUNCTION_INFO_V1(pg_llm_cross_entropy_backward);
PG_FUNCTION_INFO_V1(pg_llm_mlp_backward);

Datum
pg_llm_gelu_backward(PG_FUNCTION_ARGS)
{
    bytea *x_b = PG_GETARG_BYTEA_P(0);
    bytea *dy_b= PG_GETARG_BYTEA_P(1);
    int n;
    bytea *out;
    float *x;
    float *dy;
    float *dx;
    const float k = 0.79788456f;

    ensure_same_size(x_b, dy_b, "pg_llm_gelu_backward");

    n = float_length(x_b, "pg_llm_gelu_backward");
    (void) float_length(dy_b, "pg_llm_gelu_backward");

    if (n == 0)
        ereport(ERROR,
                (errmsg("pg_llm_gelu_backward requires non-empty tensors")));

    out = bytea_same_size(x_b);

    x = as_float(x_b);
    dy= as_float(dy_b);
    dx= as_float(out);
    for(int i=0;i<n;++i){
        float x3=x[i]*x[i]*x[i];
        float tanh_arg = k*(x[i]+0.044715f*x3);
        float sech2 = 1.0f - powf(tanhf(tanh_arg),2.0f);
        float term = 0.5f*(1.0f + tanhf(tanh_arg) + x[i]*sech2*k*(1+3*0.044715f*x[i]*x[i]));
        dx[i] = dy[i]*term;
    }
    PG_RETURN_BYTEA_P(out);
}


/* ----------------------------------------------------------
 *  Softmax backward
 * ---------------------------------------------------------- */
Datum
pg_llm_softmax_backward(PG_FUNCTION_ARGS)
{
    bytea *y_b  = PG_GETARG_BYTEA_P(0);  /* output of softmax */
    bytea *dy_b = PG_GETARG_BYTEA_P(1);  /* upstream gradient */
    int n;
    bytea *out;
    float *y;
    float *dy;
    float *dx;
    float dot = 0.f;

    ensure_same_size(y_b, dy_b, "pg_llm_softmax_backward");

    n = float_length(y_b, "pg_llm_softmax_backward");
    (void) float_length(dy_b, "pg_llm_softmax_backward");

    if (n == 0)
        ereport(ERROR,
                (errmsg("pg_llm_softmax_backward requires non-empty tensors")));

    out = bytea_same_size(y_b);

    y  = as_float(y_b);
    dy = as_float(dy_b);
    dx = as_float(out);
    for (int i=0;i<n;++i)
        dot += y[i]*dy[i];

    for (int i=0;i<n;++i)
        dx[i] = y[i]*(dy[i] - dot);

    PG_RETURN_BYTEA_P(out);
}

/* ----------------------------------------------------------
 *  Dropout backward
 * ---------------------------------------------------------- */
Datum
pg_llm_dropout_backward(PG_FUNCTION_ARGS)
{
    bytea *x_b = PG_GETARG_BYTEA_P(0);
    bytea *y_b = PG_GETARG_BYTEA_P(1);
    bytea *dy_b = PG_GETARG_BYTEA_P(2);
    float4 p = PG_GETARG_FLOAT4(3);
    bool training = PG_GETARG_BOOL(4);
    int n;
    bytea *out;
    float *x;
    float *y;
    float *dy;
    float *dx;

    if (p < 0.0f || p >= 1.0f)
        ereport(ERROR,
                (errmsg("pg_llm_dropout_backward probability must be in [0, 1) (got %f)", p)));

    ensure_same_size(x_b, y_b, "pg_llm_dropout_backward");
    ensure_same_size(x_b, dy_b, "pg_llm_dropout_backward");

    n = float_length(x_b, "pg_llm_dropout_backward");
    (void) float_length(y_b, "pg_llm_dropout_backward");
    (void) float_length(dy_b, "pg_llm_dropout_backward");

    out = bytea_same_size(x_b);
    x = as_float(x_b);
    y = as_float(y_b);
    dy = as_float(dy_b);
    dx = as_float(out);

    if (!training || p == 0.0f || n == 0) {
        memcpy(dx, dy, n * sizeof(float));
        PG_RETURN_BYTEA_P(out);
    }

    const float scale = 1.0f / (1.0f - p);

    for (int i = 0; i < n; ++i) {
        if (y[i] != 0.0f || x[i] == 0.0f)
            dx[i] = dy[i] * scale;
        else
            dx[i] = 0.0f;
    }

    PG_RETURN_BYTEA_P(out);
}

/* ----------------------------------------------------------
 *  Cross-entropy backward
 * ---------------------------------------------------------- */
Datum
pg_llm_cross_entropy_backward(PG_FUNCTION_ARGS)
{
    bytea *logits_b = PG_GETARG_BYTEA_P(0);
    int target = PG_GETARG_INT32(1);

    int n = float_length(logits_b, "pg_llm_cross_entropy_backward");
    float *logits;
    bytea *out;
    float *dlogits;
    float maxv;
    float sum = 0.0f;

    if (n == 0)
        ereport(ERROR,
                (errmsg("pg_llm_cross_entropy_backward requires a non-empty logits vector")));

    if (target < 0 || target >= n)
        ereport(ERROR,
                (errmsg("pg_llm_cross_entropy_backward target index %d out of bounds", target)));

    logits = as_float(logits_b);
    out = bytea_same_size(logits_b);
    dlogits = as_float(out);

    maxv = logits[0];
    for (int i = 1; i < n; ++i)
        if (logits[i] > maxv)
            maxv = logits[i];

    for (int i = 0; i < n; ++i) {
        float ex = expf(logits[i] - maxv);
        dlogits[i] = ex;
        sum += ex;
    }

    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; ++i)
        dlogits[i] *= inv_sum;

    dlogits[target] -= 1.0f;

    PG_RETURN_BYTEA_P(out);
}

/* ----------------------------------------------------------
 *  LayerNorm backward
 * ---------------------------------------------------------- */
Datum
pg_llm_layernorm_backward(PG_FUNCTION_ARGS)
{
    bytea *x_b  = PG_GETARG_BYTEA_P(0);
    bytea *dy_b = PG_GETARG_BYTEA_P(1);
    bytea *gamma_b = PG_GETARG_BYTEA_P(2);
    float eps = PG_GETARG_FLOAT4(3);

    int n;
    float *x;
    float *dy;
    float *g;
    bytea *dx_b;
    float *dx;
    bytea *dg_b;
    float *dg;
    bytea *db_b;
    float *db;
    float mean = 0;
    float var = 0;
    float inv_std;
    float xhat[4096]; /* assume <=4k dims per token */
    float sum_dy = 0;
    float sum_dy_xhat = 0;
    TupleDesc tupdesc;
    Datum values[3];
    bool nulls[3] = {false,false,false};
    HeapTuple rettuple;

    ensure_same_size(x_b, dy_b, "pg_llm_layernorm_backward");
    ensure_same_size(x_b, gamma_b, "pg_llm_layernorm_backward");

    n = float_length(x_b, "pg_llm_layernorm_backward");
    (void) float_length(dy_b, "pg_llm_layernorm_backward");
    (void) float_length(gamma_b, "pg_llm_layernorm_backward");

    if (n == 0)
        ereport(ERROR,
                (errmsg("pg_llm_layernorm_backward requires non-empty tensors")));
    if (n > 4096)
        ereport(ERROR,
                (errmsg("pg_llm_layernorm_backward supports up to 4096 features (got %d)", n)));
    if (eps < 0.0f)
        ereport(ERROR,
                (errmsg("pg_llm_layernorm_backward requires non-negative epsilon")));

    x  = as_float(x_b);
    dy = as_float(dy_b);
    g  = as_float(gamma_b);

    dx_b = bytea_same_size(x_b);
    dx = as_float(dx_b);

    dg_b = bytea_same_size(gamma_b);
    dg = as_float(dg_b);

    db_b = bytea_same_size(gamma_b);
    db = as_float(db_b);

    /* compute mean/var */
    for(int i=0;i<n;++i) mean+=x[i];
    mean/=n;
    for(int i=0;i<n;++i){ float d=x[i]-mean; var+=d*d; }
    var/=n;
    inv_std = 1.0f/sqrtf(var+eps);

    /* normalized */
    for(int i=0;i<n;++i) xhat[i]=(x[i]-mean)*inv_std;

    /* dBeta, dGamma */
    for(int i=0;i<n;++i){
        db[i]=dy[i];
        dg[i]=dy[i]*xhat[i];
    }

    /* sums for dX */
    for(int i=0;i<n;++i){
        sum_dy += dy[i]*g[i];
        sum_dy_xhat += dy[i]*g[i]*xhat[i];
    }

    for(int i=0;i<n;++i){
        float t1 = dy[i]*g[i];
        float t2 = sum_dy/n;
        float t3 = xhat[i]*sum_dy_xhat/n;
        dx[i] = inv_std*(t1 - t2 - t3);
    }

    if (get_call_result_type(fcinfo,NULL,&tupdesc)!=TYPEFUNC_COMPOSITE)
        ereport(ERROR,(errmsg("expected composite return")));
    BlessTupleDesc(tupdesc);
    values[0]=PointerGetDatum(dx_b);
    values[1]=PointerGetDatum(dg_b);
    values[2]=PointerGetDatum(db_b);
    rettuple = heap_form_tuple(tupdesc,values,nulls);
    PG_RETURN_DATUM(HeapTupleGetDatum(rettuple));
}

/* ----------------------------------------------------------
 *  Feed-Forward MLP backward
 * ---------------------------------------------------------- */
Datum
pg_llm_mlp_backward(PG_FUNCTION_ARGS)
{
    bytea *x_b = PG_GETARG_BYTEA_P(0);
    bytea *w_fc_b = PG_GETARG_BYTEA_P(1);
    bytea *b_fc_b = PG_GETARG_BYTEA_P(2);
    bytea *w_proj_b = PG_GETARG_BYTEA_P(3);
    bytea *b_proj_b = PG_GETARG_BYTEA_P(4);
    bytea *dy_b = PG_GETARG_BYTEA_P(5);
    int T = PG_GETARG_INT32(6);
    int D = PG_GETARG_INT32(7);

    int hidden_dim;
    Size expected_x_bytes;
    Size expected_w_fc_bytes;
    Size expected_w_proj_bytes;
    Size expected_dy_bytes;
    Size b_fc_bytes;
    Size b_proj_bytes;
    bool b_fc_broadcast;
    bool b_proj_broadcast;

    float *x;
    float *w_fc;
    float *b_fc;
    float *w_proj;
    float *b_proj;
    float *dy;

    bytea *dx_b;
    float *dx;
    bytea *dw_fc_b;
    float *dw_fc;
    bytea *db_fc_b;
    float *db_fc;
    bytea *dw_proj_b;
    float *dw_proj;
    bytea *db_proj_b;
    float *db_proj;

    float *fc_pre;
    float *fc_act;
    float *dfc;

    TupleDesc tupdesc;
    Datum values[5];
    bool nulls[5] = {false, false, false, false, false};
    HeapTuple rettuple;
    const float gelu_k = 0.79788456f;

    if (T <= 0 || D <= 0)
        ereport(ERROR,
                (errmsg("pg_llm_mlp_backward requires positive T and D")));

    hidden_dim = 4 * D;
    expected_x_bytes = (Size) T * D * sizeof(float);
    expected_w_fc_bytes = (Size) D * hidden_dim * sizeof(float);
    expected_w_proj_bytes = (Size) hidden_dim * D * sizeof(float);
    expected_dy_bytes = expected_x_bytes;

    if (nbytes(x_b) != expected_x_bytes)
        ereport(ERROR,
                (errmsg("pg_llm_mlp_backward expected x with %zu bytes (got %zu)",
                        expected_x_bytes, nbytes(x_b))));
    if (nbytes(w_fc_b) != expected_w_fc_bytes)
        ereport(ERROR,
                (errmsg("pg_llm_mlp_backward expected w_fc with %zu bytes (got %zu)",
                        expected_w_fc_bytes, nbytes(w_fc_b))));
    if (nbytes(w_proj_b) != expected_w_proj_bytes)
        ereport(ERROR,
                (errmsg("pg_llm_mlp_backward expected w_proj with %zu bytes (got %zu)",
                        expected_w_proj_bytes, nbytes(w_proj_b))));
    if (nbytes(dy_b) != expected_dy_bytes)
        ereport(ERROR,
                (errmsg("pg_llm_mlp_backward expected dy with %zu bytes (got %zu)",
                        expected_dy_bytes, nbytes(dy_b))));

    b_fc_bytes = nbytes(b_fc_b);
    b_proj_bytes = nbytes(b_proj_b);

    if (b_fc_bytes == (Size) hidden_dim * sizeof(float))
        b_fc_broadcast = true;
    else if (b_fc_bytes == (Size) T * hidden_dim * sizeof(float))
        b_fc_broadcast = false;
    else
        ereport(ERROR,
                (errmsg("pg_llm_mlp_backward expected b_fc with %zu or %zu bytes (got %zu)",
                        (Size) hidden_dim * sizeof(float),
                        (Size) T * hidden_dim * sizeof(float),
                        b_fc_bytes)));

    if (b_proj_bytes == (Size) D * sizeof(float))
        b_proj_broadcast = true;
    else if (b_proj_bytes == (Size) T * D * sizeof(float))
        b_proj_broadcast = false;
    else
        ereport(ERROR,
                (errmsg("pg_llm_mlp_backward expected b_proj with %zu or %zu bytes (got %zu)",
                        (Size) D * sizeof(float),
                        (Size) T * D * sizeof(float),
                        b_proj_bytes)));

    x = as_float(x_b);
    w_fc = as_float(w_fc_b);
    b_fc = as_float(b_fc_b);
    w_proj = as_float(w_proj_b);
    b_proj = as_float(b_proj_b);
    dy = as_float(dy_b);

    dx_b = bytea_same_size(x_b);
    dx = as_float(dx_b);
    memset(dx, 0, expected_x_bytes);

    dw_fc_b = bytea_same_size(w_fc_b);
    dw_fc = as_float(dw_fc_b);
    memset(dw_fc, 0, expected_w_fc_bytes);

    db_fc_b = bytea_same_size(b_fc_b);
    db_fc = as_float(db_fc_b);
    memset(db_fc, 0, b_fc_bytes);

    dw_proj_b = bytea_same_size(w_proj_b);
    dw_proj = as_float(dw_proj_b);
    memset(dw_proj, 0, expected_w_proj_bytes);

    db_proj_b = bytea_same_size(b_proj_b);
    db_proj = as_float(db_proj_b);
    memset(db_proj, 0, b_proj_bytes);

    fc_pre = (float *) palloc((Size) T * hidden_dim * sizeof(float));
    fc_act = (float *) palloc((Size) T * hidden_dim * sizeof(float));
    dfc = (float *) palloc((Size) T * hidden_dim * sizeof(float));

    /* Forward recomputation */
    for (int t = 0; t < T; ++t)
    {
        float *pre_row = fc_pre + (Size) t * hidden_dim;
        float *act_row = fc_act + (Size) t * hidden_dim;
        const float *x_row = x + (Size) t * D;
        for (int h = 0; h < hidden_dim; ++h)
        {
            float sum = b_fc_broadcast ? b_fc[h] : b_fc[(Size) t * hidden_dim + h];
            for (int d = 0; d < D; ++d)
                sum += x_row[d] * w_fc[(Size) d * hidden_dim + h];
            pre_row[h] = sum;
            float x3 = sum * sum * sum;
            act_row[h] = 0.5f * sum * (1.0f + tanhf(gelu_k * (sum + 0.044715f * x3)));
        }
    }

    /* Gradients w.r.t. projection layer */
    for (int t = 0; t < T; ++t)
    {
        const float *dy_row = dy + (Size) t * D;
        if (!b_proj_broadcast)
        {
            float *db_row = db_proj + (Size) t * D;
            memcpy(db_row, dy_row, (Size) D * sizeof(float));
        }
        for (int j = 0; j < D; ++j)
        {
            float grad = dy_row[j];
            if (b_proj_broadcast)
                db_proj[j] += grad;
        }
    }

    for (int h = 0; h < hidden_dim; ++h)
    {
        const float *w_proj_row = w_proj + (Size) h * D;
        for (int j = 0; j < D; ++j)
        {
            float accum = 0.0f;
            for (int t = 0; t < T; ++t)
            {
                const float *act_row = fc_act + (Size) t * hidden_dim;
                const float *dy_row = dy + (Size) t * D;
                accum += act_row[h] * dy_row[j];
            }
            dw_proj[(Size) h * D + j] = accum;
        }
    }

    for (int t = 0; t < T; ++t)
    {
        const float *dy_row = dy + (Size) t * D;
        float *dfc_row = dfc + (Size) t * hidden_dim;
        for (int h = 0; h < hidden_dim; ++h)
        {
            const float *w_proj_row = w_proj + (Size) h * D;
            float accum = 0.0f;
            for (int j = 0; j < D; ++j)
                accum += dy_row[j] * w_proj_row[j];
            dfc_row[h] = accum;
        }
    }

    /* Backprop through GELU */
    for (int t = 0; t < T; ++t)
    {
        float *dfc_row = dfc + (Size) t * hidden_dim;
        float *pre_row = fc_pre + (Size) t * hidden_dim;
        for (int h = 0; h < hidden_dim; ++h)
        {
            float xval = pre_row[h];
            float x2 = xval * xval;
            float x3 = x2 * xval;
            float tanh_arg = gelu_k * (xval + 0.044715f * x3);
            float tanh_val = tanhf(tanh_arg);
            float sech2 = 1.0f - tanh_val * tanh_val;
            float term = 0.5f * (1.0f + tanh_val + xval * sech2 * gelu_k * (1.0f + 3.0f * 0.044715f * x2));
            dfc_row[h] *= term;
        }
    }

    /* Gradients for b_fc */
    if (b_fc_broadcast)
    {
        for (int h = 0; h < hidden_dim; ++h)
        {
            float sum = 0.0f;
            for (int t = 0; t < T; ++t)
                sum += dfc[(Size) t * hidden_dim + h];
            db_fc[h] = sum;
        }
    }
    else
    {
        memcpy(db_fc, dfc, (Size) T * hidden_dim * sizeof(float));
    }

    /* Gradients for w_fc */
    for (int d = 0; d < D; ++d)
    {
        for (int h = 0; h < hidden_dim; ++h)
        {
            float accum = 0.0f;
            for (int t = 0; t < T; ++t)
            {
                const float *x_row = x + (Size) t * D;
                const float *dfc_row = dfc + (Size) t * hidden_dim;
                accum += x_row[d] * dfc_row[h];
            }
            dw_fc[(Size) d * hidden_dim + h] = accum;
        }
    }

    /* Gradients for x */
    for (int t = 0; t < T; ++t)
    {
        float *dx_row = dx + (Size) t * D;
        const float *dfc_row = dfc + (Size) t * hidden_dim;
        for (int d = 0; d < D; ++d)
        {
            float accum = 0.0f;
            for (int h = 0; h < hidden_dim; ++h)
                accum += dfc_row[h] * w_fc[(Size) d * hidden_dim + h];
            dx_row[d] = accum;
        }
    }

    if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
        ereport(ERROR, (errmsg("expected composite return")));
    BlessTupleDesc(tupdesc);

    values[0] = PointerGetDatum(dx_b);
    values[1] = PointerGetDatum(dw_fc_b);
    values[2] = PointerGetDatum(db_fc_b);
    values[3] = PointerGetDatum(dw_proj_b);
    values[4] = PointerGetDatum(db_proj_b);

    rettuple = heap_form_tuple(tupdesc, values, nulls);
    PG_RETURN_DATUM(HeapTupleGetDatum(rettuple));
}
