#include "pg_llm.h"

/*
 * Multi-head self-attention
 * args:
 *   x:    BYTEA (T×D)
 *   w_qkv: BYTEA (D×3D)
 *   w_o:   BYTEA (D×D)
 *   n_head: INT
 * returns: BYTEA (T×D)
 */
PG_FUNCTION_INFO_V1(pg_llm_attention);
Datum pg_llm_attention(PG_FUNCTION_ARGS)
{
    bytea *x_b    = PG_GETARG_BYTEA_P(0);
    bytea *w_qkvb = PG_GETARG_BYTEA_P(1);
    bytea *w_ob   = PG_GETARG_BYTEA_P(2);
    int n_head    = PG_GETARG_INT32(3);

    const int64_t T = PG_GETARG_INT32(4);  /* sequence length */
    const int64_t D = PG_GETARG_INT32(5);  /* model dim */

    float *x    = as_float(x_b);
    float *w_qkv= as_float(w_qkvb);
    float *w_o  = as_float(w_ob);

    /* allocate temporary buffers */
    const int head_dim = D / n_head;
    const float scale = 1.0f / sqrtf((float)head_dim);
    bytea *out;
    float *Y;
    float *Q;
    float *K;
    float *V;
    float *tmp;

    out = (bytea*) palloc(T*D*sizeof(float) + VARHDRSZ);
    SET_VARSIZE(out, T*D*sizeof(float) + VARHDRSZ);
    Y = as_float(out);

    /* 1. Q,K,V = x @ W_qkv, then split */
    Q = palloc(T * D * sizeof(float));
    K = palloc(T * D * sizeof(float));
    V = palloc(T * D * sizeof(float));
    for (int t=0; t<T; ++t) {
        for (int j=0; j<3*D; ++j) {
            float s = 0.0f;
            for (int k=0; k<D; ++k)
                s += x[t*D + k] * w_qkv[k*3*D + j];
            if (j < D)
                Q[t*D + j] = s;
            else if (j < 2*D)
                K[t*D + (j - D)] = s;
            else
                V[t*D + (j - 2*D)] = s;
        }
    }

    /* 2. iterate heads */
    for (int h=0; h<n_head; ++h) {
        int off = h*head_dim;
        /* compute attention weights: (T×head_dim) × (head_dim×T) */
        for (int i=0; i<T; ++i) {
            int q_base = i*D + off;
            float scores[1024]; /* max ctx len = 1024 */
            float maxs = -1e9f;
            float sum;
            for (int j=0;j<=i;++j){          /* causal mask */
                int k_base = j*D + off;
                float dot = 0;
                for(int d=0;d<head_dim;++d)
                    dot += Q[q_base + d]*K[k_base + d];
                dot *= scale;
                scores[j]=dot;
                if(dot>maxs)maxs=dot;
            }
            /* softmax */
            sum = 0;
            for(int j=0;j<=i;++j){ scores[j]=expf(scores[j]-maxs); sum+=scores[j]; }
            for(int j=0;j<=i;++j) scores[j]/=sum;

            /* weighted sum over V */
            for(int d=0;d<head_dim;++d){
                float acc=0;
                for(int j=0;j<=i;++j){
                    int v_base = j*D + off;
                    acc += scores[j]*V[v_base + d];
                }
                Y[q_base + d] = acc;
            }
        }
    }

    /* 3. project output: Y = Y @ W_o */
    tmp = palloc(T * D * sizeof(float));
    for (int i=0; i<T; ++i) {
        for (int j=0; j<D; ++j){
            float s = 0;
            for (int k=0;k<D;++k)
                s += Y[i*D + k]*w_o[k*D + j];
            tmp[i*D + j]=s;
        }
    }
    memcpy(Y,tmp,T*D*sizeof(float));
    pfree(tmp);
    pfree(Q);
    pfree(K);
    pfree(V);
    PG_RETURN_BYTEA_P(out);
}
