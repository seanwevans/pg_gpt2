#include "pg_llm.h"
#include <string.h>

#if defined(__AVX__) || defined(__AVX2__)
#include <immintrin.h>
#endif

/* Tile sizes tuned for GPT-style matrices (e.g. 1024x768). */
#define TILE_M 64
#define TILE_N 64

#if defined(__AVX__) || defined(__AVX2__)
static inline float
hsum256_ps(__m256 v)
{
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehl_ps(vlow, vlow);
    vlow = _mm_add_ps(vlow, shuf);
    shuf = _mm_shuffle_ps(vlow, vlow, 0x55);
    vlow = _mm_add_ss(vlow, shuf);
    return _mm_cvtss_f32(vlow);
}
#endif

void
pg_llm_fast_gemm(const float *A, const float *B, float *C,
                 int M, int K, int N)
{
    if (M <= 0 || K <= 0 || N <= 0)
        return;

    for (int j0 = 0; j0 < N; j0 += TILE_N) {
        int n_block = Min(TILE_N, N - j0);
        Size tile_bytes = (Size)n_block * K * sizeof(float);
        float *B_tile = (float *) palloc(tile_bytes);

        for (int jj = 0; jj < n_block; ++jj) {
            int col = j0 + jj;
            const float *src = B + col;
            float *dst = B_tile + (Size)jj * K;
            for (int kk = 0; kk < K; ++kk)
                dst[kk] = src[(Size)kk * N];
        }

        for (int i0 = 0; i0 < M; i0 += TILE_M) {
            int m_block = Min(TILE_M, M - i0);
            for (int i = 0; i < m_block; ++i) {
                const float *a_row = A + (Size)(i0 + i) * K;
                float *c_row = C + (Size)(i0 + i) * N + j0;
                for (int jj = 0; jj < n_block; ++jj) {
                    const float *b_col = B_tile + (Size)jj * K;
#if defined(__AVX2__)
                    __m256 acc = _mm256_setzero_ps();
                    int k = 0;
                    for (; k <= K - 8; k += 8) {
                        __m256 av = _mm256_loadu_ps(a_row + k);
                        __m256 bv = _mm256_loadu_ps(b_col + k);
                        acc = _mm256_fmadd_ps(av, bv, acc);
                    }
                    float sum = hsum256_ps(acc);
                    for (; k < K; ++k)
                        sum += a_row[k] * b_col[k];
#elif defined(__AVX__)
                    __m256 acc = _mm256_setzero_ps();
                    int k = 0;
                    for (; k <= K - 8; k += 8) {
                        __m256 av = _mm256_loadu_ps(a_row + k);
                        __m256 bv = _mm256_loadu_ps(b_col + k);
                        acc = _mm256_add_ps(acc, _mm256_mul_ps(av, bv));
                    }
                    float sum = hsum256_ps(acc);
                    for (; k < K; ++k)
                        sum += a_row[k] * b_col[k];
#else
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k)
                        sum += a_row[k] * b_col[k];
#endif
                    c_row[jj] = sum;
                }
            }
        }
        pfree(B_tile);
    }
}
