#ifndef PG_LLM_H
#define PG_LLM_H

#include "postgres.h"
#include "fmgr.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include <math.h>

PG_MODULE_MAGIC;

/* Function declarations */
PG_FUNCTION_INFO_V1(pg_llm_matmul);
PG_FUNCTION_INFO_V1(pg_llm_add);
PG_FUNCTION_INFO_V1(pg_llm_gelu);
PG_FUNCTION_INFO_V1(pg_llm_softmax);
PG_FUNCTION_INFO_V1(pg_llm_layernorm);
PG_FUNCTION_INFO_V1(pg_llm_cross_entropy);

/* Utility helpers */
static inline float* as_float(bytea *b) {
    return (float*) VARDATA_ANY(b);
}
static inline size_t nbytes(bytea *b) {
    return VARSIZE_ANY_EXHDR(b);
}

#endif
