#ifndef PG_LLM_H
#define PG_LLM_H

#include "postgres.h"
#include "fmgr.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "funcapi.h"
#include "executor/spi.h"
#include <math.h>

/* Function declarations */
extern Datum pg_llm_matmul(PG_FUNCTION_ARGS);
extern Datum pg_llm_add(PG_FUNCTION_ARGS);
extern Datum pg_llm_gelu(PG_FUNCTION_ARGS);
extern Datum pg_llm_softmax(PG_FUNCTION_ARGS);
extern Datum pg_llm_layernorm(PG_FUNCTION_ARGS);
extern Datum pg_llm_cross_entropy(PG_FUNCTION_ARGS);
extern Datum pg_llm_dropout(PG_FUNCTION_ARGS);
extern Datum pg_llm_ones_like(PG_FUNCTION_ARGS);
extern Datum pg_llm_zeros_like(PG_FUNCTION_ARGS);
extern Datum pg_llm_transpose(PG_FUNCTION_ARGS);
extern Datum pg_llm_attention(PG_FUNCTION_ARGS);
extern Datum pg_llm_attention_backward(PG_FUNCTION_ARGS);
extern Datum pg_llm_gelu_backward(PG_FUNCTION_ARGS);
extern Datum pg_llm_softmax_backward(PG_FUNCTION_ARGS);
extern Datum pg_llm_layernorm_backward(PG_FUNCTION_ARGS);
extern Datum pg_llm_dropout_backward(PG_FUNCTION_ARGS);
extern Datum pg_llm_export_npz(PG_FUNCTION_ARGS);
extern Datum pg_llm_cross_entropy_backward(PG_FUNCTION_ARGS);
extern Datum pg_llm_attention_backward(PG_FUNCTION_ARGS);
extern Datum pg_llm_mlp_backward(PG_FUNCTION_ARGS);

/* Autograd instrumentation helpers */
extern bool pg_llm_autograd_enabled(void);
extern int pg_llm_autograd_track_tensor(bytea *tensor, int ndims, const int *dims, bool requires_grad);
extern void pg_llm_autograd_record_tape(const char *name, int *inputs, int n_inputs, int output, const char *extra_json);
extern Datum pg_llm_autograd_map_param(PG_FUNCTION_ARGS);

/* Optimized kernels */
extern void pg_llm_fast_gemm(const float *A, const float *B, float *C,
                             int M, int K, int N);
extern void pg_llm_vector_add(const float *a, const float *b, float *out, int n);
extern void pg_llm_layernorm_forward(const float *x,
                                     const float *gamma,
                                     const float *beta,
                                     int n,
                                     float eps,
                                     float *y);

/* Utility helpers */
static inline float* as_float(bytea *b) {
    return (float*) VARDATA_ANY(b);
}

static inline size_t nbytes(bytea *b) {
    return VARSIZE_ANY_EXHDR(b);
}

static inline bytea* bytea_alloc(Size payload_bytes) {
    bytea *out = (bytea*) palloc(payload_bytes + VARHDRSZ);
    SET_VARSIZE(out, payload_bytes + VARHDRSZ);
    return out;
}

static inline bytea* bytea_same_size(bytea *src) {
    return bytea_alloc(nbytes(src));
}

static inline int float_length(bytea *b, const char *fn_name) {
    Size size = nbytes(b);
    if (size % sizeof(float) != 0)
        ereport(ERROR,
                (errmsg("%s expected a float32-aligned bytea (got %zu bytes)",
                        fn_name, size)));
    return (int)(size / sizeof(float));
}

static inline bytea* bytea_constant_like(bytea *src, const char *fn_name, float value) {
    int n = float_length(src, fn_name);
    bytea *out = bytea_same_size(src);
    float *dst = as_float(out);
    for (int i = 0; i < n; i++)
        dst[i] = value;
    return out;
}

static inline void ensure_same_size(bytea *a, bytea *b, const char *fn_name) {
    Size size_a = nbytes(a);
    Size size_b = nbytes(b);
    if (size_a != size_b) {
        ereport(ERROR,
                (errmsg("%s expects inputs with identical length (got %zu and %zu bytes)",
                        fn_name, size_a, size_b)));
    }
}

#endif
