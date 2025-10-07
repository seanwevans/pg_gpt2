#include "pg_llm.h"
#include <stdlib.h>

extern Datum drandom(PG_FUNCTION_ARGS);

static int
compare_desc_float(const void *a, const void *b)
{
    float x = *(const float *)a;
    float y = *(const float *)b;

    if (x < y)
        return 1;
    if (x > y)
        return -1;
    return 0;
}

PG_FUNCTION_INFO_V1(pg_llm_sample);

/*
 * pg_llm_sample(logits BYTEA, temperature FLOAT4, topk INT, topp FLOAT4)
 * Returns INT (selected token index)
 */
Datum pg_llm_sample(PG_FUNCTION_ARGS)
{
    bytea *z_b = PG_GETARG_BYTEA_P(0);
    float  temp = PG_GETARG_FLOAT4(1);
    int    topk = PG_GETARG_INT32(2);
    float  topp = PG_GETARG_FLOAT4(3);

    int n = (int)(nbytes(z_b)/sizeof(float));
    float *z = as_float(z_b);
    float *p;
    float maxv;
    float sum = 0;
    float r;
    float c = 0;
    int chosen = n - 1;
    FmgrInfo flinfo;
    LOCAL_FCINFO(random_fcinfo, 0);

    if (temp <= 0) temp = 1.0f;
    p = palloc(n*sizeof(float));

    /* 1. softmax with temperature */
    maxv = z[0]/temp;
    for (int i=1;i<n;++i) if (z[i]/temp > maxv) maxv = z[i]/temp;
    for (int i=0;i<n;++i){ p[i]=expf((z[i]/temp)-maxv); sum+=p[i]; }
    for (int i=0;i<n;++i) p[i]/=sum;

    /* 2. top-k pruning */
    if (topk>0 && topk<n) {
        float *copy = palloc(n*sizeof(float));
        float thresh;
        float s = 0;
        memcpy(copy,p,n*sizeof(float));
        qsort(copy,n,sizeof(float),compare_desc_float);
        thresh = copy[topk-1];
        for(int i=0;i<n;++i) if(p[i]<thresh) p[i]=0;
        pfree(copy);
        for(int i=0;i<n;++i) s+=p[i];
        if(s>0) for(int i=0;i<n;++i) p[i]/=s;
    }

    /* 3. top-p pruning */
    if (topp>0 && topp<1.0f) {
        int *idx = palloc(n*sizeof(int));
        float cum = 0;
        float s = 0;
        for(int i=0;i<n;++i) idx[i]=i;
        /* sort indices by prob desc */
        for(int i=0;i<n-1;++i)
            for(int j=i+1;j<n;++j)
                if(p[idx[j]]>p[idx[i]]){int t=idx[i];idx[i]=idx[j];idx[j]=t;}
        for(int k=0;k<n;++k){
            cum+=p[idx[k]];
            if(cum>topp){
                for(int j=k+1;j<n;++j) p[idx[j]]=0;
                break;
            }
        }
        for(int i=0;i<n;++i)s+=p[i];
        if(s>0)for(int i=0;i<n;++i)p[i]/=s;
        pfree(idx);
    }

    /* 4. sample */
    flinfo.fn_addr = drandom;
    flinfo.fn_oid = InvalidOid;
    flinfo.fn_nargs = 0;
    flinfo.fn_strict = false;
    flinfo.fn_retset = false;
    flinfo.fn_stats = 0;
    flinfo.fn_extra = NULL;
    flinfo.fn_mcxt = CurrentMemoryContext;
    flinfo.fn_expr = NULL;

    InitFunctionCallInfoData(*random_fcinfo, &flinfo, 0, InvalidOid, NULL, NULL);

    random_fcinfo->isnull = false;
    r = (float) DatumGetFloat8(FunctionCallInvoke(random_fcinfo));
    for(int i=0;i<n;++i){ c+=p[i]; if(r<=c){chosen=i;break;} }

    pfree(p);

    PG_RETURN_INT32(chosen);
}
