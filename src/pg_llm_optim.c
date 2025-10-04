#include "pg_llm.h"

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

    int n = (int)(nbytes(w_b) / sizeof(float));

    bytea *w_out = (bytea*) palloc(n*sizeof(float) + VARHDRSZ);
    bytea *m_out = (bytea*) palloc(n*sizeof(float) + VARHDRSZ);
    bytea *v_out = (bytea*) palloc(n*sizeof(float) + VARHDRSZ);
    SET_VARSIZE(w_out, n*sizeof(float) + VARHDRSZ);
    SET_VARSIZE(m_out, n*sizeof(float) + VARHDRSZ);
    SET_VARSIZE(v_out, n*sizeof(float) + VARHDRSZ);

    float *w = as_float(w_b);
    float *g = as_float(g_b);
    float *m = as_float(m_b);
    float *v = as_float(v_b);
    float *wo= as_float(w_out);
    float *mo= as_float(m_out);
    float *vo= as_float(v_out);

    /* bias-correction */
    float bc1 = 1.0f - powf(b1, t);
    float bc2 = 1.0f - powf(b2, t);

    for (int i=0;i<n;++i) {
        float grad = g[i] + wd * w[i];           /* decoupled weight decay */
        float m_t  = b1*m[i] + (1.0f-b1)*grad;
        float v_t  = b2*v[i] + (1.0f-b2)*grad*grad;
        float m_hat = m_t / bc1;
        float v_hat = v_t / bc2;
        float step  = lr * m_hat / (sqrtf(v_hat) + eps);
        wo[i] = w[i] - step;
        mo[i] = m_t;
        vo[i] = v_t;
    }

    TupleDesc tupdesc;
    Datum values[3];
    bool nulls[3] = {false,false,false};

    if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
        ereport(ERROR, (errmsg("expected composite return type")));

    BlessTupleDesc(tupdesc);
    values[0] = PointerGetDatum(w_out);
    values[1] = PointerGetDatum(m_out);
    values[2] = PointerGetDatum(v_out);

    HeapTuple rettuple = heap_form_tuple(tupdesc, values, nulls);
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
    int n = (int)(nbytes(g_b)/sizeof(float));
    float *g = as_float(g_b);

    float norm=0;
    for(int i=0;i<n;++i) norm += g[i]*g[i];
    norm = sqrtf(norm);
    float scale = (norm>clip)? (clip/norm) : 1.0f;

    bytea *out = (bytea*) palloc(n*sizeof(float)+VARHDRSZ);
    SET_VARSIZE(out, n*sizeof(float)+VARHDRSZ);
    float *go = as_float(out);
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
    if (step < warmup)
        lr = lr_max * (float)step / (float)warmup;
    else {
        float progress = (float)(step - warmup) / (float)(total - warmup);
        lr = 0.5f * lr_max * (1.0f + cosf(M_PI * progress));
    }
    PG_RETURN_FLOAT4(lr);
}
