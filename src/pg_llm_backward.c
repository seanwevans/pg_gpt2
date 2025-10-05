#include "pg_llm.h"
#include <string.h>

PG_FUNCTION_INFO_V1(pg_llm_gelu_backward);
PG_FUNCTION_INFO_V1(pg_llm_softmax_backward);
PG_FUNCTION_INFO_V1(pg_llm_layernorm_backward);
PG_FUNCTION_INFO_V1(pg_llm_dropout_backward);
PG_FUNCTION_INFO_V1(pg_llm_cross_entropy_backward);

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

    n = (int)(nbytes(x_b)/sizeof(float));
    out = (bytea*) palloc(n*sizeof(float)+VARHDRSZ);
    SET_VARSIZE(out,n*sizeof(float)+VARHDRSZ);

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
    int n = (int)(nbytes(y_b)/sizeof(float));
    bytea *out;
    float *y;
    float *dy;
    float *dx;
    float dot = 0.f;

    out = (bytea*) palloc(n*sizeof(float)+VARHDRSZ);
    SET_VARSIZE(out, n*sizeof(float)+VARHDRSZ);

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

    int n = (int)(nbytes(x_b)/sizeof(float));
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

    x  = as_float(x_b);
    dy = as_float(dy_b);
    g  = as_float(gamma_b);

    dx_b = (bytea*) palloc(n*sizeof(float)+VARHDRSZ);
    SET_VARSIZE(dx_b,n*sizeof(float)+VARHDRSZ);
    dx = as_float(dx_b);

    dg_b = (bytea*) palloc(n*sizeof(float)+VARHDRSZ);
    SET_VARSIZE(dg_b,n*sizeof(float)+VARHDRSZ);
    dg = as_float(dg_b);

    db_b = (bytea*) palloc(n*sizeof(float)+VARHDRSZ);
    SET_VARSIZE(db_b,n*sizeof(float)+VARHDRSZ);
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
