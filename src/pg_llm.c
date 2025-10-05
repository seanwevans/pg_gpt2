#include "pg_llm.h"

extern Datum drandom(PG_FUNCTION_ARGS);

PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(pg_llm_matmul);
PG_FUNCTION_INFO_V1(pg_llm_add);
PG_FUNCTION_INFO_V1(pg_llm_gelu);
PG_FUNCTION_INFO_V1(pg_llm_softmax);
PG_FUNCTION_INFO_V1(pg_llm_layernorm);
PG_FUNCTION_INFO_V1(pg_llm_cross_entropy);
PG_FUNCTION_INFO_V1(pg_llm_dropout);

/* ------------------ MATMUL ------------------ */
Datum pg_llm_matmul(PG_FUNCTION_ARGS)
{
    bytea *a = PG_GETARG_BYTEA_P(0);
    bytea *b = PG_GETARG_BYTEA_P(1);
    int m = PG_GETARG_INT32(2);
    int k = PG_GETARG_INT32(3);
    int n = PG_GETARG_INT32(4);

    size_t expected_a;
    size_t expected_b;
    size_t out_bytes;
    bytea *out;
    float *A;
    float *B;
    float *C;

    if (m <= 0 || k <= 0 || n <= 0)
        ereport(ERROR, (errmsg("pg_llm_matmul requires positive matrix dimensions")));

    expected_a = (size_t) m * k * sizeof(float);
    expected_b = (size_t) k * n * sizeof(float);
    if (nbytes(a) != expected_a)
        ereport(ERROR,
                (errmsg("pg_llm_matmul expected left matrix of %d x %d elements", m, k)));
    if (nbytes(b) != expected_b)
        ereport(ERROR,
                (errmsg("pg_llm_matmul expected right matrix of %d x %d elements", k, n)));

    A = as_float(a);
    B = as_float(b);
    out_bytes = (size_t) m * n * sizeof(float);
    out = bytea_alloc(out_bytes);
    C = as_float(out);

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
    int n;
    bytea *out;
    float *A;
    float *B;
    float *C;

    ensure_same_size(a, b, "pg_llm_add");

    n = float_length(a, "pg_llm_add");
    (void) float_length(b, "pg_llm_add");

    out = bytea_same_size(a);

    A = as_float(a);
    B = as_float(b);
    C = as_float(out);

    for (int i = 0; i < n; i++)
        C[i] = A[i] + B[i];

    PG_RETURN_BYTEA_P(out);
}

/* ------------------ GELU ------------------ */
/* tanh approximation: 0.5 * x * (1 + tanh(√(2/π)*(x + 0.044715*x³))) */
Datum pg_llm_gelu(PG_FUNCTION_ARGS)
{
    bytea *a = PG_GETARG_BYTEA_P(0);
    int n = float_length(a, "pg_llm_gelu");
    bytea *out;
    float *A;
    float *Y;
    const float k = 0.79788456f;   // √(2/π)
    if (n == 0)
        ereport(ERROR, (errmsg("pg_llm_gelu requires a non-empty input")));

    out = bytea_same_size(a);

    A = as_float(a);
    Y = as_float(out);

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
    int n = float_length(a, "pg_llm_softmax");
    float *A;
    bytea *out;
    float *Y;
    float maxv;
    float sum = 0.0f;
    if (n == 0)
        ereport(ERROR, (errmsg("pg_llm_softmax requires a non-empty input")));

    A = as_float(a);

    out = bytea_same_size(a);
    Y = as_float(out);

    maxv = A[0];
    for (int i = 1; i < n; i++) if (A[i] > maxv) maxv = A[i];

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

    int n = float_length(x_b, "pg_llm_layernorm");
    float *x;
    float *gamma;
    float *beta;
    bytea *out;
    float *y;
    float mean = 0.f;
    float var = 0.f;
    float inv_std;
    if (n == 0)
        ereport(ERROR, (errmsg("pg_llm_layernorm requires a non-empty input")));

    x = as_float(x_b);
    gamma = as_float(gamma_b);
    beta = as_float(beta_b);

    (void) float_length(gamma_b, "pg_llm_layernorm");
    (void) float_length(beta_b, "pg_llm_layernorm");
    ensure_same_size(x_b, gamma_b, "pg_llm_layernorm");
    ensure_same_size(x_b, beta_b, "pg_llm_layernorm");

    out = bytea_same_size(x_b);
    y = as_float(out);

    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;

    for (int i = 0; i < n; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= n;
    inv_std = 1.0f / sqrtf(var + eps);

    for (int i = 0; i < n; i++)
        y[i] = ((x[i] - mean) * inv_std) * gamma[i] + beta[i];

    PG_RETURN_BYTEA_P(out);
}

/* ------------------ CROSS ENTROPY ------------------ */
Datum pg_llm_cross_entropy(PG_FUNCTION_ARGS)
{
    bytea *logits_b = PG_GETARG_BYTEA_P(0);
    int target = PG_GETARG_INT32(1);

    int n = float_length(logits_b, "pg_llm_cross_entropy");
    float *z;
    float maxv;
    float sum = 0.0f;
    float log_sum;
    float loss;
    if (n == 0)
        ereport(ERROR, (errmsg("pg_llm_cross_entropy requires a non-empty logits vector")));

    z = as_float(logits_b);

    if (target < 0 || target >= n)
        ereport(ERROR, (errmsg("pg_llm_cross_entropy target index %d out of bounds", target)));

    maxv = z[0];
    for (int i = 1; i < n; i++) if (z[i] > maxv) maxv = z[i];

    for (int i = 0; i < n; i++)
        sum += expf(z[i] - maxv);
    log_sum = logf(sum) + maxv;

    loss = log_sum - z[target];  /* −log softmax[target] */
    PG_RETURN_FLOAT4(loss);
}

/* ------------------ DROPOUT ------------------ */
Datum pg_llm_dropout(PG_FUNCTION_ARGS)
{
    bytea *input = PG_GETARG_BYTEA_P(0);
    float4 p = PG_GETARG_FLOAT4(1);
    bool training = PG_GETARG_BOOL(2);
    int n;
    bytea *out;
    float *src;
    float *dst;
    float scale;
    FmgrInfo flinfo;
    LOCAL_FCINFO(random_fcinfo, 0);

    if (p < 0.0f || p >= 1.0f)
        ereport(ERROR,
                (errmsg("pg_llm_dropout probability must be in [0, 1) (got %f)", p)));

    n = float_length(input, "pg_llm_dropout");

    if (!training || p == 0.0f || n == 0)
        PG_RETURN_BYTEA_P(input);

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

    out = bytea_same_size(input);
    src = as_float(input);
    dst = as_float(out);
    scale = 1.0f / (1.0f - p);

    for (int i = 0; i < n; i++) {
        Datum rand_datum;
        double r;

        random_fcinfo->isnull = false;
        rand_datum = FunctionCallInvoke(random_fcinfo);
        r = DatumGetFloat8(rand_datum);
        if (r < p)
            dst[i] = 0.0f;
        else
            dst[i] = src[i] * scale;
    }

    PG_RETURN_BYTEA_P(out);
}
