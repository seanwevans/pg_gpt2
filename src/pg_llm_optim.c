#include "pg_llm.h"
#include <limits.h>

/* ----------------------------------------------------------
 *  AdamW: parameter update
 *  args:
 *     weight, grad, m, v : BYTEA buffers of float32, equal size
 *     lr, beta1, beta2, eps, wd : FLOAT4
 *     t : INT  (step number)
 *  returns:
 *     record(weight_out BYTEA, m_out BYTEA, v_out BYTEA)
 * ---------------------------------------------------------- */

PG_FUNCTION_INFO_V1(pg_llm_adamw_step);
Datum pg_llm_adamw_step(PG_FUNCTION_ARGS)
{
    bytea *w_b  = PG_GETARG_BYTEA_P(0);
    bytea *g_b  = PG_GETARG_BYTEA_P(1);
    bytea *m_b  = PG_GETARG_BYTEA_P(2);
    bytea *v_b  = PG_GETARG_BYTEA_P(3);
    float  lr   = PG_GETARG_FLOAT4(4);
    float  b1   = PG_GETARG_FLOAT4(5);
    float  b2   = PG_GETARG_FLOAT4(6);
    float  eps  = PG_GETARG_FLOAT4(7);
    float  wd   = PG_GETARG_FLOAT4(8);
    int    t    = PG_GETARG_INT32(9);

    Size  w_bytes = nbytes(w_b);
    Size  g_bytes = nbytes(g_b);
    Size  m_bytes = nbytes(m_b);
    Size  v_bytes = nbytes(v_b);
    int   n;
    bytea *w_out;
    bytea *m_out;
    bytea *v_out;
    float *w;
    float *g;
    float *m;
    float *v;
    float *wo;
    float *mo;
    float *vo;
    float bc1;
    float bc2;
    TupleDesc tupdesc;
    Datum values[3];
    bool nulls[3] = {false,false,false};
    HeapTuple rettuple;

    if (w_bytes == 0)
        ereport(ERROR,
                (errmsg("pg_llm_adamw_step requires non-empty tensors")));
    if (w_bytes % sizeof(float) != 0)
        ereport(ERROR,
                (errmsg("pg_llm_adamw_step weight buffer must be float32 aligned")));
    if (w_bytes != g_bytes || w_bytes != m_bytes || w_bytes != v_bytes)
        ereport(ERROR,
                (errmsg("pg_llm_adamw_step expects all tensors to have the same length")));
    if (w_bytes / sizeof(float) > INT_MAX)
        ereport(ERROR,
                (errmsg("pg_llm_adamw_step tensor length exceeds INT_MAX")));

    n = (int) (w_bytes / sizeof(float));

    if (t <= 0)
        ereport(ERROR,
                (errmsg("pg_llm_adamw_step requires positive step number (got %d)", t)));

    w_out = bytea_same_size(w_b);
    m_out = bytea_same_size(m_b);
    v_out = bytea_same_size(v_b);

    w = as_float(w_b);
    g = as_float(g_b);
    m = as_float(m_b);
    v = as_float(v_b);
    wo= as_float(w_out);
    mo= as_float(m_out);
    vo= as_float(v_out);

    /* bias-correction */
    bc1 = 1.0f - powf(b1, t);
    bc2 = 1.0f - powf(b2, t);
    if (bc1 == 0.0f || bc2 == 0.0f)
        ereport(ERROR,
                (errmsg("pg_llm_adamw_step encountered zero bias correction term")));

    /* Follow AdamW (Loshchilov & Hutter, 2019) with decoupled weight decay. */
    for (int i=0;i<n;++i) {
        float grad = g[i];
        float m_t  = b1*m[i] + (1.0f-b1)*grad;
        float v_t  = b2*v[i] + (1.0f-b2)*grad*grad;
        float m_hat = m_t / bc1;
        float v_hat = v_t / bc2;
        float step  = lr * m_hat / (sqrtf(v_hat) + eps);
        wo[i] = w[i] - step - lr * wd * w[i];
        mo[i] = m_t;
        vo[i] = v_t;
    }

    if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
        ereport(ERROR, (errmsg("expected composite return type")));

    BlessTupleDesc(tupdesc);
    values[0] = PointerGetDatum(w_out);
    values[1] = PointerGetDatum(m_out);
    values[2] = PointerGetDatum(v_out);

    rettuple = heap_form_tuple(tupdesc, values, nulls);
    PG_RETURN_DATUM(HeapTupleGetDatum(rettuple));
}

/* ----------------------------------------------------------
 *  Gradient-clipping by global norm
 * ---------------------------------------------------------- */
PG_FUNCTION_INFO_V1(pg_llm_grad_clip);
Datum pg_llm_grad_clip(PG_FUNCTION_ARGS)
{
    bytea *g_b = PG_GETARG_BYTEA_P(0);
    float  clip = PG_GETARG_FLOAT4(1);
    Size bytes = nbytes(g_b);
    int n;
    float *g;
    float norm=0;
    float scale;
    bytea *out;
    float *go;

    if (bytes == 0)
        ereport(ERROR,
                (errmsg("pg_llm_grad_clip requires a non-empty gradient tensor")));
    if (bytes % sizeof(float) != 0)
        ereport(ERROR,
                (errmsg("pg_llm_grad_clip gradient tensor must be float32 aligned")));
    if (bytes / sizeof(float) > INT_MAX)
        ereport(ERROR,
                (errmsg("pg_llm_grad_clip tensor length exceeds INT_MAX")));
    if (clip < 0.0f)
        ereport(ERROR,
                (errmsg("pg_llm_grad_clip requires non-negative clip value")));

    n = (int) (bytes / sizeof(float));
    g = as_float(g_b);

    for(int i=0;i<n;++i) norm += g[i]*g[i];
    norm = sqrtf(norm);
    scale = (norm>clip)? (clip/norm) : 1.0f;

    out = bytea_same_size(g_b);
    go = as_float(out);
    for(int i=0;i<n;++i) go[i] = g[i]*scale;

    PG_RETURN_BYTEA_P(out);
}

/* ----------------------------------------------------------
 *  Cosine-decay LR schedule with warm-up
 * ---------------------------------------------------------- */
PG_FUNCTION_INFO_V1(pg_llm_lr_schedule);
Datum pg_llm_lr_schedule(PG_FUNCTION_ARGS)
{
    int step = PG_GETARG_INT32(0);
    int warmup = PG_GETARG_INT32(1);
    int total  = PG_GETARG_INT32(2);
    float lr_max = PG_GETARG_FLOAT4(3);

    float lr;
    if (warmup <= 0)
        ereport(ERROR,
                (errmsg("pg_llm_lr_schedule warmup steps must be positive")));
    if (total <= warmup)
        ereport(ERROR,
                (errmsg("pg_llm_lr_schedule total steps must exceed warmup steps")));
    if (lr_max < 0.0f)
        ereport(ERROR,
                (errmsg("pg_llm_lr_schedule requires non-negative lr_max")));

    if (step < warmup)
        lr = lr_max * (float)step / (float)warmup;
    else {
        float progress = (float)(step - warmup) / (float)(total - warmup);
        lr = 0.5f * lr_max * (1.0f + cosf(M_PI * progress));
    }
    PG_RETURN_FLOAT4(lr);
}
