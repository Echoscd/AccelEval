#include <stdlib.h>
#include <string.h>

static int g_ni = 0, g_nj = 0, g_nk = 0;
static float g_alpha = 0.0f, g_beta = 0.0f;
static const float *g_A = NULL;
static const float *g_B = NULL;
static const float *g_C0 = NULL;
static float *g_work = NULL;

void solution_init(int ni, int nj, int nk,
                   int alpha_milli, int beta_milli,
                   const float *A, const float *B, const float *C0) {
    g_ni = ni;
    g_nj = nj;
    g_nk = nk;
    g_alpha = ((float)alpha_milli) / 1000.0f;
    g_beta = ((float)beta_milli) / 1000.0f;
    g_A = A;
    g_B = B;
    g_C0 = C0;
    g_work = (float*)malloc((size_t)ni * (size_t)nj * sizeof(float));
}

void solution_compute(float *C_out) {
    size_t total = (size_t)g_ni * (size_t)g_nj;
    memcpy(g_work, g_C0, total * sizeof(float));

    for (int i = 0; i < g_ni; ++i) {
        float *Crow = g_work + (size_t)i * (size_t)g_nj;
        for (int j = 0; j < g_nj; ++j) {
            Crow[j] *= g_beta;
        }
        const float *Arow = g_A + (size_t)i * (size_t)g_nk;
        for (int k = 0; k < g_nk; ++k) {
            float aik = g_alpha * Arow[k];
            const float *Brow = g_B + (size_t)k * (size_t)g_nj;
            for (int j = 0; j < g_nj; ++j) {
                Crow[j] += aik * Brow[j];
            }
        }
    }

    memcpy(C_out, g_work, total * sizeof(float));
}

void solution_free(void) {
    free(g_work);
    g_work = NULL;
    g_A = NULL;
    g_B = NULL;
    g_C0 = NULL;
    g_ni = g_nj = g_nk = 0;
    g_alpha = g_beta = 0.0f;
}
