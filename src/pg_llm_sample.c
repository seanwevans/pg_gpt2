#include "pg_llm.h"
#include <stdlib.h>
#include <time.h>

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

    if (temp <= 0) temp = 1.0f;
    float *p = palloc(n*sizeof(float));

    /* 1. softmax with temperature */
    float maxv = z[0]/temp;
    for (int i=1;i<n;++i) if (z[i]/temp > maxv) maxv = z[i]/temp;
    float sum=0;
    for (int i=0;i<n;++i){ p[i]=expf((z[i]/temp)-maxv); sum+=p[i]; }
    for (int i=0;i<n;++i) p[i]/=sum;

    /* 2. top-k pruning */
    if (topk>0 && topk<n) {
        float *copy = palloc(n*sizeof(float));
        memcpy(copy,p,n*sizeof(float));
        qsort(copy,n,sizeof(float),[](const void*a,const void*b){
            float x=*(float*)a, y=*(float*)b; return (y>x)-(y<x);
        });
        float thresh = copy[topk-1];
        for(int i=0;i<n;++i) if(p[i]<thresh) p[i]=0;
        pfree(copy);
        float s=0; for(int i=0;i<n;++i) s+=p[i];
        if(s>0) for(int i=0;i<n;++i) p[i]/=s;
    }

    /* 3. top-p pruning */
    if (topp>0 && topp<1.0f) {
        int *idx = palloc(n*sizeof(int));
        for(int i=0;i<n;++i) idx[i]=i;
        /* sort indices by prob desc */
        for(int i=0;i<n-1;++i)
            for(int j=i+1;j<n;++j)
                if(p[idx[j]]>p[idx[i]]){int t=idx[i];idx[i]=idx[j];idx[j]=t;}
        float cum=0;
        for(int k=0;k<n;++k){
            cum+=p[idx[k]];
            if(cum>topp){
                for(int j=k+1;j<n;++j) p[idx[j]]=0;
                break;
            }
        }
        float s=0;for(int i=0;i<n;++i)s+=p[i];
        if(s>0)for(int i=0;i<n;++i)p[i]/=s;
        pfree(idx);
    }

    /* 4. sample */
    float r = (float)rand()/(float)RAND_MAX;
    float c=0;
    int chosen=n-1;
    for(int i=0;i<n;++i){ c+=p[i]; if(r<=c){chosen=i;break;} }

    PG_RETURN_INT32(chosen);
}
