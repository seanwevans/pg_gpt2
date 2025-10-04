#include "pg_llm.h"

/* ------------------ MATMUL ------------------ */
Datum pg_llm_matmul(PG_FUNCTION_ARGS)
{
    bytea *a = PG_GETARG_BYTEA_P(0);
    bytea *b = PG_GETARG_BYTEA_P(1);
    int m = PG_GETARG_INT32(2);
    int k = PG_GETARG_INT32(3);
    int n = PG_GETARG_INT32(4);

    float *A = as_float(a);
    float *B = as_float(b);
    size_t out_bytes = m * n * sizeof(float);
    bytea *out = (bytea*) palloc(out_bytes + VARHDRSZ);
    SET_VARSIZE(out, out_bytes + VARHDRSZ);
    float *C = as_float(out);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int t = 0; t < k; t++)
                sum += A[i*k + t] * B[t*n + j];
            C[i*n + j] = sum;
        }
    }
    PG_RETURN_BYTEA_P(out);
}

/* ------------------ ADD ------------------ */
Datum pg_llm_add(PG_FUNCTION_ARGS)
{
    bytea *a = PG_GETARG_BYTEA_P(0);
    bytea *b = PG_GETARG_BYTEA_P(1);
    int n = (int)(nbytes(a) / sizeof(float));

    bytea *out = (bytea*) palloc(VARSIZE_ANY_EXHDR(a) + VARHDRSZ);
    SET_VARSIZE(out, VARSIZE_ANY_EXHDR(a) + VARHDRSZ);

    float *A = as_float(a);
    float *B = as_float(b);
    float *C = as_float(out);

    for (int i = 0; i < n; i++)
        C[i] = A[i] + B[i];

    PG_RETURN_BYTEA_P(out);
}

/* ------------------ GELU ------------------ */
/* tanh approximation: 0.5 * x * (1 + tanh(√(2/π)*(x + 0.044715*x³))) */
Datum pg_llm_gelu(PG_FUNCTION_ARGS)
{
    bytea *a = PG_GETARG_BYTEA_P(0);
    int n = (int)(nbytes(a) / sizeof(float));

    bytea *out = (bytea*) palloc(VARSIZE_ANY_EXHDR(a) + VARHDRSZ);
    SET_VARSIZE(out, VARSIZE_ANY_EXHDR(a) + VARHDRSZ);

    float *A = as_float(a);
    float *Y = as_float(out);

    const float k = 0.79788456f;   // √(2/π)
    for (int i = 0; i < n; i++) {
        float x = A[i];
        float x3 = x * x * x;
        Y[i] = 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x3)));
    }

    PG_RETURN_BYTEA_P(out);
}

/* ------------------ SOFTMAX ------------------ */
Datum pg_llm_softmax(PG_FUNCTION_ARGS)
{
    bytea *a = PG_GETARG_BYTEA_P(0);
    int n = (int)(nbytes(a) / sizeof(float));
    float *A = as_float(a);

    bytea *out = (bytea*) palloc(VARSIZE_ANY_EXHDR(a) + VARHDRSZ);
    SET_VARSIZE(out, VARSIZE_ANY_EXHDR(a) + VARHDRSZ);
    float *Y = as_float(out);

    float maxv = A[0];
    for (int i = 1; i < n; i++) if (A[i] > maxv) maxv = A[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        Y[i] = expf(A[i] - maxv);
        sum += Y[i];
    }
    for (int i = 0; i < n; i++)
        Y[i] /= sum;

    PG_RETURN_BYTEA_P(out);
}

/* ------------------ LAYER NORM ------------------ */
Datum pg_llm_layernorm(PG_FUNCTION_ARGS)
{
    bytea *x_b = PG_GETARG_BYTEA_P(0);
    bytea *gamma_b = PG_GETARG_BYTEA_P(1);
    bytea *beta_b = PG_GETARG_BYTEA_P(2);
    float eps = PG_GETARG_FLOAT4(3);

    int n = (int)(nbytes(x_b) / sizeof(float));
    float *x = as_float(x_b);
    float *gamma = as_float(gamma_b);
    float *beta = as_float(beta_b);

    bytea *out = (bytea*) palloc(VARSIZE_ANY_EXHDR(x_b) + VARHDRSZ);
    SET_VARSIZE(out, VARSIZE_ANY_EXHDR(x_b) + VARHDRSZ);
    float *y = as_float(out);

    float mean = 0.f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;

    float var = 0.f;
    for (int i = 0; i < n; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= n;
    float inv_std = 1.0f / sqrtf(var + eps);

    for (int i = 0; i < n; i++)
        y[i] = ((x[i] - mean) * inv_std) * gamma[i] + beta[i];

    PG_RETURN_BYTEA_P(out);
}

/* ------------------ CROSS ENTROPY ------------------ */
Datum pg_llm_cross_entropy(PG_FUNCTION_ARGS)
{
    bytea *logits_b = PG_GETARG_BYTEA_P(0);
    int target = PG_GETARG_INT32(1);

    int n = (int)(nbytes(logits_b) / sizeof(float));
    float *z = as_float(logits_b);

    float maxv = z[0];
    for (int i = 1; i < n; i++) if (z[i] > maxv) maxv = z[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++)
        sum += expf(z[i] - maxv);
    float log_sum = logf(sum) + maxv;

    float loss = log_sum - z[target];  /* −log softmax[target] */
    PG_RETURN_FLOAT4(loss);
}
