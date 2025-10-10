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
PG_FUNCTION_INFO_V1(pg_llm_ones_like);
PG_FUNCTION_INFO_V1(pg_llm_zeros_like);
PG_FUNCTION_INFO_V1(pg_llm_transpose);

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
    bool autograd = pg_llm_autograd_enabled();

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

    if (autograd)
    {
        int dims_a[2] = {m, k};
        int dims_b[2] = {k, n};

        if (SPI_connect() != SPI_OK_CONNECT)
            ereport(ERROR, (errmsg("SPI_connect failed in pg_llm_matmul")));

        PG_TRY();
        {
            int input_ids[2];
            input_ids[0] = pg_llm_autograd_track_tensor(a, 2, dims_a, true);
            input_ids[1] = pg_llm_autograd_track_tensor(b, 2, dims_b, true);
            pg_llm_fast_gemm(A, B, C, m, k, n);
            int dims_out[2] = {m, n};
            int output_id = pg_llm_autograd_track_tensor(out, 2, dims_out, true);
            char *extra = psprintf("{\"m\":%d,\"k\":%d,\"n\":%d}", m, k, n);
            pg_llm_autograd_record_tape("matmul", input_ids, 2, output_id, extra);
        }
        PG_CATCH();
        {
            SPI_finish();
            PG_RE_THROW();
        }
        PG_END_TRY();

        SPI_finish();
    }
    else
    {
        pg_llm_fast_gemm(A, B, C, m, k, n);
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
    bool autograd = pg_llm_autograd_enabled();

    ensure_same_size(a, b, "pg_llm_add");

    n = float_length(a, "pg_llm_add");
    (void) float_length(b, "pg_llm_add");

    out = bytea_same_size(a);

    A = as_float(a);
    B = as_float(b);
    C = as_float(out);

    if (autograd)
    {
        if (SPI_connect() != SPI_OK_CONNECT)
            ereport(ERROR, (errmsg("SPI_connect failed in pg_llm_add")));

        PG_TRY();
        {
            int dims[1] = {n};
            int input_ids[2];
            input_ids[0] = pg_llm_autograd_track_tensor(a, 1, dims, true);
            input_ids[1] = pg_llm_autograd_track_tensor(b, 1, dims, true);
            pg_llm_vector_add(A, B, C, n);
            int output_id = pg_llm_autograd_track_tensor(out, 1, dims, true);
            pg_llm_autograd_record_tape("add", input_ids, 2, output_id, NULL);
        }
        PG_CATCH();
        {
            SPI_finish();
            PG_RE_THROW();
        }
        PG_END_TRY();

        SPI_finish();
    }
    else
    {
        pg_llm_vector_add(A, B, C, n);
    }

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
    bool autograd = pg_llm_autograd_enabled();
    if (n == 0)
        ereport(ERROR, (errmsg("pg_llm_gelu requires a non-empty input")));

    out = bytea_same_size(a);

    A = as_float(a);
    Y = as_float(out);

    if (autograd)
    {
        if (SPI_connect() != SPI_OK_CONNECT)
            ereport(ERROR, (errmsg("SPI_connect failed in pg_llm_gelu")));

        PG_TRY();
        {
            int dims[1] = {n};
            int input_ids[1];
            input_ids[0] = pg_llm_autograd_track_tensor(a, 1, dims, true);
            for (int i = 0; i < n; i++)
            {
                float x = A[i];
                float x3 = x * x * x;
                Y[i] = 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x3)));
            }
            int output_id = pg_llm_autograd_track_tensor(out, 1, dims, true);
            pg_llm_autograd_record_tape("gelu", input_ids, 1, output_id, NULL);
        }
        PG_CATCH();
        {
            SPI_finish();
            PG_RE_THROW();
        }
        PG_END_TRY();

        SPI_finish();
    }
    else
    {
        for (int i = 0; i < n; i++) {
            float x = A[i];
            float x3 = x * x * x;
            Y[i] = 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x3)));
        }
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
    bool autograd = pg_llm_autograd_enabled();
    if (n == 0)
        ereport(ERROR, (errmsg("pg_llm_softmax requires a non-empty input")));

    A = as_float(a);

    out = bytea_same_size(a);
    Y = as_float(out);

    if (autograd)
    {
        if (SPI_connect() != SPI_OK_CONNECT)
            ereport(ERROR, (errmsg("SPI_connect failed in pg_llm_softmax")));

        PG_TRY();
        {
            int dims[1] = {n};
            int input_ids[1];
            input_ids[0] = pg_llm_autograd_track_tensor(a, 1, dims, true);
            maxv = A[0];
            for (int i = 1; i < n; i++) if (A[i] > maxv) maxv = A[i];

            for (int i = 0; i < n; i++) {
                Y[i] = expf(A[i] - maxv);
                sum += Y[i];
            }
            for (int i = 0; i < n; i++)
                Y[i] /= sum;
            int output_id = pg_llm_autograd_track_tensor(out, 1, dims, true);
            pg_llm_autograd_record_tape("softmax", input_ids, 1, output_id, NULL);
        }
        PG_CATCH();
        {
            SPI_finish();
            PG_RE_THROW();
        }
        PG_END_TRY();

        SPI_finish();
    }
    else
    {
        maxv = A[0];
        for (int i = 1; i < n; i++) if (A[i] > maxv) maxv = A[i];

        for (int i = 0; i < n; i++) {
            Y[i] = expf(A[i] - maxv);
            sum += Y[i];
        }
        for (int i = 0; i < n; i++)
            Y[i] /= sum;
    }

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
    bool autograd = pg_llm_autograd_enabled();
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

    if (autograd)
    {
        if (SPI_connect() != SPI_OK_CONNECT)
            ereport(ERROR, (errmsg("SPI_connect failed in pg_llm_layernorm")));

        PG_TRY();
        {
            int dims[1] = {n};
            int input_ids[3];
            input_ids[0] = pg_llm_autograd_track_tensor(x_b, 1, dims, true);
            input_ids[1] = pg_llm_autograd_track_tensor(gamma_b, 1, dims, true);
            input_ids[2] = pg_llm_autograd_track_tensor(beta_b, 1, dims, true);

            pg_llm_layernorm_forward(x, gamma, beta, n, eps, y);

            int output_id = pg_llm_autograd_track_tensor(out, 1, dims, true);
            char *extra = psprintf("{\"eps\":%.9g,\"gamma_id\":%d,\"beta_id\":%d}",
                                    eps, input_ids[1], input_ids[2]);
            pg_llm_autograd_record_tape("layernorm", input_ids, 3, output_id, extra);
        }
        PG_CATCH();
        {
            SPI_finish();
            PG_RE_THROW();
        }
        PG_END_TRY();

        SPI_finish();
    }
    else
    {
        pg_llm_layernorm_forward(x, gamma, beta, n, eps, y);
    }

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
    bool autograd = pg_llm_autograd_enabled();
    if (n == 0)
        ereport(ERROR, (errmsg("pg_llm_cross_entropy requires a non-empty logits vector")));

    z = as_float(logits_b);

    if (target < 0 || target >= n)
        ereport(ERROR, (errmsg("pg_llm_cross_entropy target index %d out of bounds", target)));

    if (autograd)
    {
        if (SPI_connect() != SPI_OK_CONNECT)
            ereport(ERROR, (errmsg("SPI_connect failed in pg_llm_cross_entropy")));

        PG_TRY();
        {
            int dims_in[1] = {n};
            int input_ids[1];
            input_ids[0] = pg_llm_autograd_track_tensor(logits_b, 1, dims_in, true);

            maxv = z[0];
            for (int i = 1; i < n; i++) if (z[i] > maxv) maxv = z[i];

            sum = 0.0f;
            for (int i = 0; i < n; i++)
                sum += expf(z[i] - maxv);
            log_sum = logf(sum) + maxv;

            loss = log_sum - z[target];

            bytea *loss_buf = bytea_alloc(sizeof(float));
            float *loss_ptr = as_float(loss_buf);
            loss_ptr[0] = loss;
            int dims_out[1] = {1};
            int output_id = pg_llm_autograd_track_tensor(loss_buf, 1, dims_out, true);
            char *extra = psprintf("{\"target\":%d}", target);
            pg_llm_autograd_record_tape("cross_entropy", input_ids, 1, output_id, extra);
        }
        PG_CATCH();
        {
            SPI_finish();
            PG_RE_THROW();
        }
        PG_END_TRY();

        SPI_finish();
    }
    else
    {
        maxv = z[0];
        for (int i = 1; i < n; i++) if (z[i] > maxv) maxv = z[i];

        for (int i = 0; i < n; i++)
            sum += expf(z[i] - maxv);
        log_sum = logf(sum) + maxv;

        loss = log_sum - z[target];  /* −log softmax[target] */
    }

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
    bool autograd = pg_llm_autograd_enabled();

    if (p < 0.0f || p >= 1.0f)
        ereport(ERROR,
                (errmsg("pg_llm_dropout probability must be in [0, 1) (got %f)", p)));

    n = float_length(input, "pg_llm_dropout");

    if (!training || p == 0.0f || n == 0)
    {
        if (autograd)
        {
            if (SPI_connect() != SPI_OK_CONNECT)
                ereport(ERROR, (errmsg("SPI_connect failed in pg_llm_dropout")));

            PG_TRY();
            {
                int dims[1] = {n};
                int input_ids[1];
                input_ids[0] = pg_llm_autograd_track_tensor(input, 1, dims, true);
                char *extra = psprintf("{\"p\":%.6f,\"training\":false}", p);
                pg_llm_autograd_record_tape("dropout", input_ids, 1, input_ids[0], extra);
            }
            PG_CATCH();
            {
                SPI_finish();
                PG_RE_THROW();
            }
            PG_END_TRY();

            SPI_finish();
        }
        PG_RETURN_BYTEA_P(input);
    }

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

    if (autograd)
    {
        if (SPI_connect() != SPI_OK_CONNECT)
            ereport(ERROR, (errmsg("SPI_connect failed in pg_llm_dropout")));

        PG_TRY();
        {
            int dims[1] = {n};
            int input_ids[1];
            input_ids[0] = pg_llm_autograd_track_tensor(input, 1, dims, true);
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
            int output_id = pg_llm_autograd_track_tensor(out, 1, dims, true);
            char *extra = psprintf("{\"p\":%.6f,\"training\":true}", p);
            pg_llm_autograd_record_tape("dropout", input_ids, 1, output_id, extra);
        }
        PG_CATCH();
        {
            SPI_finish();
            PG_RE_THROW();
        }
        PG_END_TRY();

        SPI_finish();
    }
    else
    {
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
    }

    PG_RETURN_BYTEA_P(out);
}

/* ------------------ TENSOR HELPERS ------------------ */
Datum pg_llm_ones_like(PG_FUNCTION_ARGS)
{
    bytea *input = PG_GETARG_BYTEA_P(0);
    bytea *out;
    bool autograd = pg_llm_autograd_enabled();
    int n = float_length(input, "pg_llm_ones_like");

    out = bytea_constant_like(input, "pg_llm_ones_like", 1.0f);

    if (autograd)
    {
        if (SPI_connect() != SPI_OK_CONNECT)
            ereport(ERROR, (errmsg("SPI_connect failed in pg_llm_ones_like")));

        PG_TRY();
        {
            int dims[1] = {n};
            int input_ids[1];
            input_ids[0] = pg_llm_autograd_track_tensor(input, 1, dims, false);
            int output_id = pg_llm_autograd_track_tensor(out, 1, dims, false);
            pg_llm_autograd_record_tape("ones_like", input_ids, 1, output_id,
                                        "{\"value\":1.0}");
        }
        PG_CATCH();
        {
            SPI_finish();
            PG_RE_THROW();
        }
        PG_END_TRY();

        SPI_finish();
    }

    PG_RETURN_BYTEA_P(out);
}

Datum pg_llm_zeros_like(PG_FUNCTION_ARGS)
{
    bytea *input = PG_GETARG_BYTEA_P(0);
    bytea *out;
    bool autograd = pg_llm_autograd_enabled();
    int n = float_length(input, "pg_llm_zeros_like");

    out = bytea_constant_like(input, "pg_llm_zeros_like", 0.0f);

    if (autograd)
    {
        if (SPI_connect() != SPI_OK_CONNECT)
            ereport(ERROR, (errmsg("SPI_connect failed in pg_llm_zeros_like")));

        PG_TRY();
        {
            int dims[1] = {n};
            int input_ids[1];
            input_ids[0] = pg_llm_autograd_track_tensor(input, 1, dims, false);
            int output_id = pg_llm_autograd_track_tensor(out, 1, dims, false);
            pg_llm_autograd_record_tape("zeros_like", input_ids, 1, output_id,
                                        "{\"value\":0.0}");
        }
        PG_CATCH();
        {
            SPI_finish();
            PG_RE_THROW();
        }
        PG_END_TRY();

        SPI_finish();
    }

    PG_RETURN_BYTEA_P(out);
}

Datum pg_llm_transpose(PG_FUNCTION_ARGS)
{
    bytea *input = PG_GETARG_BYTEA_P(0);
    int rows = PG_GETARG_INT32(1);
    int cols = PG_GETARG_INT32(2);
    size_t expected_bytes;
    bytea *out;
    float *src;
    float *dst;
    bool autograd = pg_llm_autograd_enabled();

    if (rows <= 0 || cols <= 0)
        ereport(ERROR,
                (errmsg("pg_llm_transpose requires positive dimensions (got %d x %d)",
                        rows, cols)));

    expected_bytes = (size_t) rows * cols * sizeof(float);
    if (nbytes(input) != expected_bytes)
        ereport(ERROR,
                (errmsg("pg_llm_transpose expected %d x %d matrix (%zu bytes)",
                        rows, cols, expected_bytes)));

    out = bytea_alloc(expected_bytes);
    src = as_float(input);
    dst = as_float(out);

    if (autograd)
    {
        if (SPI_connect() != SPI_OK_CONNECT)
            ereport(ERROR, (errmsg("SPI_connect failed in pg_llm_transpose")));

        PG_TRY();
        {
            int dims_in[2] = {rows, cols};
            int dims_out[2] = {cols, rows};
            int input_ids[1];
            input_ids[0] = pg_llm_autograd_track_tensor(input, 2, dims_in, true);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    dst[j * rows + i] = src[i * cols + j];
            int output_id = pg_llm_autograd_track_tensor(out, 2, dims_out, true);
            char *extra = psprintf("{\"rows\":%d,\"cols\":%d}", rows, cols);
            pg_llm_autograd_record_tape("transpose", input_ids, 1, output_id, extra);
        }
        PG_CATCH();
        {
            SPI_finish();
            PG_RE_THROW();
        }
        PG_END_TRY();

        SPI_finish();
    }
    else
    {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                dst[j * rows + i] = src[i * cols + j];
    }

    PG_RETURN_BYTEA_P(out);
}
