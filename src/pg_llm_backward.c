#include "pg_llm.h"

PG_FUNCTION_INFO_V1(pg_llm_gelu_backward);
Datum pg_llm_gelu_backward(PG_FUNCTION_ARGS)
{
    bytea *x_b = PG_GETARG_BYTEA_P(0);
    bytea *dy_b= PG_GETARG_BYTEA_P(1);
    int n = (int)(nbytes(x_b)/sizeof(float));
    bytea *out = (bytea*) palloc(n*sizeof(float)+VARHDRSZ);
    SET_VARSIZE(out,n*sizeof(float)+VARHDRSZ);

    float *x = as_float(x_b);
    float *dy= as_float(dy_b);
    float *dx= as_float(out);

    const float k = 0.79788456f;
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
PG_FUNCTION_INFO_V1(pg_llm_softmax_backward);
Datum pg_llm_softmax_backward(PG_FUNCTION_ARGS)
{
    bytea *y_b  = PG_GETARG_BYTEA_P(0);  /* output of softmax */
    bytea *dy_b = PG_GETARG_BYTEA_P(1);  /* upstream gradient */
    int n = (int)(nbytes(y_b)/sizeof(float));

    bytea *out = (bytea*) palloc(n*sizeof(float)+VARHDRSZ);
    SET_VARSIZE(out, n*sizeof(float)+VARHDRSZ);

    float *y  = as_float(y_b);
    float *dy = as_float(dy_b);
    float *dx = as_float(out);

    float dot = 0.f;
    for (int i=0;i<n;++i)
        dot += y[i]*dy[i];

    for (int i=0;i<n;++i)
        dx[i] = y[i]*(dy[i] - dot);

    PG_RETURN_BYTEA_P(out);
}

/* ----------------------------------------------------------
 *  LayerNorm backward
 * ---------------------------------------------------------- */
PG_FUNCTION_INFO_V1(pg_llm_layernorm_backward);
Datum pg_llm_layernorm_backward(PG_FUNCTION_ARGS)
{
    bytea *x_b  = PG_GETARG_BYTEA_P(0);
    bytea *dy_b = PG_GETARG_BYTEA_P(1);
    bytea *gamma_b = PG_GETARG_BYTEA_P(2);
    float eps = PG_GETARG_FLOAT4(3);

    int n = (int)(nbytes(x_b)/sizeof(float));
    float *x  = as_float(x_b);
    float *dy = as_float(dy_b);
    float *g  = as_float(gamma_b);

    bytea *dx_b = (bytea*) palloc(n*sizeof(float)+VARHDRSZ);
    SET_VARSIZE(dx_b,n*sizeof(float)+VARHDRSZ);
    float *dx = as_float(dx_b);

    bytea *dg_b = (bytea*) palloc(n*sizeof(float)+VARHDRSZ);
    SET_VARSIZE(dg_b,n*sizeof(float)+VARHDRSZ);
    float *dg = as_float(dg_b);

    bytea *db_b = (bytea*) palloc(n*sizeof(float)+VARHDRSZ);
    SET_VARSIZE(db_b,n*sizeof(float)+VARHDRSZ);
    float *db = as_float(db_b);

    /* compute mean/var */
    float mean=0, var=0;
    for(int i=0;i<n;++i) mean+=x[i];
    mean/=n;
    for(int i=0;i<n;++i){ float d=x[i]-mean; var+=d*d; }
    var/=n;
    float inv_std = 1.0f/sqrtf(var+eps);

    /* normalized */
    float xhat[4096]; /* assume <=4k dims per token */
    for(int i=0;i<n;++i) xhat[i]=(x[i]-mean)*inv_std;

    /* dBeta, dGamma */
    for(int i=0;i<n;++i){
        db[i]=dy[i];
        dg[i]=dy[i]*xhat[i];
    }

    /* sums for dX */
    float sum_dy=0, sum_dy_xhat=0;
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

    TupleDesc tupdesc;
    Datum values[3];
    bool nulls[3] = {false,false,false};
    if (get_call_result_type(fcinfo,NULL,&tupdesc)!=TYPEFUNC_COMPOSITE)
        ereport(ERROR,(errmsg("expected composite return")));
    BlessTupleDesc(tupdesc);
    values[0]=PointerGetDatum(dx_b);
    values[1]=PointerGetDatum(dg_b);
    values[2]=PointerGetDatum(db_b);
    HeapTuple rettuple = heap_form_tuple(tupdesc,values,nulls);
    PG_RETURN_DATUM(HeapTupleGetDatum(rettuple));
}
