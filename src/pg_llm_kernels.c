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

    int max_n_block = Min(TILE_N, N);
    Size tile_bytes = (Size) max_n_block * K * sizeof(float);
    float *B_tile = (float *) palloc(tile_bytes);

    for (int j0 = 0; j0 < N; j0 += TILE_N) {
        int n_block = Min(TILE_N, N - j0);

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
    }

    pfree(B_tile);
}

void
pg_llm_vector_add(const float *a, const float *b, float *out, int n)
{
#if defined(__AVX__) || defined(__AVX2__)
    int i = 0;
    for (; i <= n - 8; i += 8)
    {
        __m256 av = _mm256_loadu_ps(a + i);
        __m256 bv = _mm256_loadu_ps(b + i);
        __m256 sum = _mm256_add_ps(av, bv);
        _mm256_storeu_ps(out + i, sum);
    }
    for (; i < n; ++i)
        out[i] = a[i] + b[i];
#else
    for (int i = 0; i < n; ++i)
        out[i] = a[i] + b[i];
#endif
}

void
pg_llm_layernorm_forward(const float *x,
                         const float *gamma,
                         const float *beta,
                         int n,
                         float eps,
                         float *y)
{
    float sum = 0.0f;
    float sumsq = 0.0f;
    int i = 0;

#if defined(__AVX__) || defined(__AVX2__)
    __m256 sum_v = _mm256_setzero_ps();
    __m256 sumsq_v = _mm256_setzero_ps();
    for (; i <= n - 8; i += 8)
    {
        __m256 xv = _mm256_loadu_ps(x + i);
        sum_v = _mm256_add_ps(sum_v, xv);
#if defined(__AVX2__) && defined(__FMA__)
        sumsq_v = _mm256_fmadd_ps(xv, xv, sumsq_v);
#else
        __m256 xv_sq = _mm256_mul_ps(xv, xv);
        sumsq_v = _mm256_add_ps(sumsq_v, xv_sq);
#endif
    }
    sum += hsum256_ps(sum_v);
    sumsq += hsum256_ps(sumsq_v);
#endif

    for (; i < n; ++i)
    {
        float xv = x[i];
        sum += xv;
        sumsq += xv * xv;
    }

    float mean = sum / (float) n;
    float var = sumsq / (float) n - mean * mean;
    if (var < 0.0f)
        var = 0.0f;
    float inv_std = 1.0f / sqrtf(var + eps);

    i = 0;
#if defined(__AVX__) || defined(__AVX2__)
    __m256 mean_v = _mm256_set1_ps(mean);
    __m256 inv_std_v = _mm256_set1_ps(inv_std);
    for (; i <= n - 8; i += 8)
    {
        __m256 xv = _mm256_loadu_ps(x + i);
        __m256 gv = _mm256_loadu_ps(gamma + i);
        __m256 bv = _mm256_loadu_ps(beta + i);
        __m256 norm = _mm256_mul_ps(_mm256_sub_ps(xv, mean_v), inv_std_v);
#if defined(__AVX2__) && defined(__FMA__)
        __m256 yv = _mm256_fmadd_ps(norm, gv, bv);
#else
        __m256 yv = _mm256_add_ps(_mm256_mul_ps(norm, gv), bv);
#endif
        _mm256_storeu_ps(y + i, yv);
    }
#endif

    for (; i < n; ++i)
    {
        float norm = (x[i] - mean) * inv_std;
        y[i] = norm * gamma[i] + beta[i];
    }
}
