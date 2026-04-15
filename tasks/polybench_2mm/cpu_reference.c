#include <stdlib.h>
#include <string.h>

static int g_ni = 0, g_nj = 0, g_nk = 0, g_nl = 0;
static float g_alpha = 0.0f, g_beta = 0.0f;
static const float *g_A = NULL;
static const float *g_B = NULL;
static const float *g_C = NULL;
static const float *g_D0 = NULL;
static float *g_tmp = NULL;
static float *g_work = NULL;

void solution_init(int ni, int nj, int nk, int nl,
                   int alpha_milli, int beta_milli,
                   const float *A, const float *B,
                   const float *C, const float *D0) {
    g_ni = ni;
    g_nj = nj;
    g_nk = nk;
    g_nl = nl;
    g_alpha = ((float)alpha_milli) / 1000.0f;
    g_beta = ((float)beta_milli) / 1000.0f;
    g_A = A;
    g_B = B;
    g_C = C;
    g_D0 = D0;
    g_tmp = (float*)malloc((size_t)ni * (size_t)nj * sizeof(float));
    g_work = (float*)malloc((size_t)ni * (size_t)nl * sizeof(float));
}

void solution_compute(float *D_out) {
    for (int i = 0; i < g_ni; ++i) {
        float *tmp_row = g_tmp + (size_t)i * (size_t)g_nj;
        for (int j = 0; j < g_nj; ++j) tmp_row[j] = 0.0f;
        const float *Arow = g_A + (size_t)i * (size_t)g_nk;
        for (int k = 0; k < g_nk; ++k) {
            float aik = g_alpha * Arow[k];
            const float *Brow = g_B + (size_t)k * (size_t)g_nj;
            for (int j = 0; j < g_nj; ++j) {
                tmp_row[j] += aik * Brow[j];
            }
        }
    }

    memcpy(g_work, g_D0, (size_t)g_ni * (size_t)g_nl * sizeof(float));

    for (int i = 0; i < g_ni; ++i) {
        float *Drow = g_work + (size_t)i * (size_t)g_nl;
        const float *tmprow = g_tmp + (size_t)i * (size_t)g_nj;
        for (int j = 0; j < g_nl; ++j) {
            Drow[j] *= g_beta;
        }
        for (int k = 0; k < g_nj; ++k) {
            float tik = tmprow[k];
            const float *Crow = g_C + (size_t)k * (size_t)g_nl;
            for (int j = 0; j < g_nl; ++j) {
                Drow[j] += tik * Crow[j];
            }
        }
    }

    memcpy(D_out, g_work, (size_t)g_ni * (size_t)g_nl * sizeof(float));
}

void solution_free(void) {
    free(g_tmp);
    free(g_work);
    g_tmp = NULL;
    g_work = NULL;
    g_A = NULL;
    g_B = NULL;
    g_C = NULL;
    g_D0 = NULL;
    g_ni = g_nj = g_nk = g_nl = 0;
    g_alpha = g_beta = 0.0f;
}
