#include <stdlib.h>

static int g_ni = 0;
static int g_nj = 0;
static int g_nk = 0;
static int g_nl = 0;
static int g_nm = 0;
static const float *g_A = NULL;
static const float *g_B = NULL;
static const float *g_C = NULL;
static const float *g_D = NULL;

void solution_init(int ni, int nj, int nk, int nl, int nm,
                   const float *A, const float *B,
                   const float *C, const float *D) {
    g_ni = ni;
    g_nj = nj;
    g_nk = nk;
    g_nl = nl;
    g_nm = nm;
    g_A = A;
    g_B = B;
    g_C = C;
    g_D = D;
}

void solution_compute(float *G_out) {
    const size_t e_size = (size_t)g_ni * (size_t)g_nj;
    const size_t f_size = (size_t)g_nj * (size_t)g_nl;
    float *E = (float*)malloc(e_size * sizeof(float));
    float *F = (float*)malloc(f_size * sizeof(float));
    if (!E || !F) {
        free(E);
        free(F);
        return;
    }

    for (int i = 0; i < g_ni; ++i) {
        const size_t e_row = (size_t)i * (size_t)g_nj;
        const size_t a_row = (size_t)i * (size_t)g_nk;
        for (int j = 0; j < g_nj; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < g_nk; ++k) {
                acc += g_A[a_row + (size_t)k] * g_B[(size_t)k * (size_t)g_nj + (size_t)j];
            }
            E[e_row + (size_t)j] = acc;
        }
    }

    for (int i = 0; i < g_nj; ++i) {
        const size_t f_row = (size_t)i * (size_t)g_nl;
        const size_t c_row = (size_t)i * (size_t)g_nm;
        for (int j = 0; j < g_nl; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < g_nm; ++k) {
                acc += g_C[c_row + (size_t)k] * g_D[(size_t)k * (size_t)g_nl + (size_t)j];
            }
            F[f_row + (size_t)j] = acc;
        }
    }

    for (int i = 0; i < g_ni; ++i) {
        const size_t g_row = (size_t)i * (size_t)g_nl;
        const size_t e_row = (size_t)i * (size_t)g_nj;
        for (int j = 0; j < g_nl; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < g_nj; ++k) {
                acc += E[e_row + (size_t)k] * F[(size_t)k * (size_t)g_nl + (size_t)j];
            }
            G_out[g_row + (size_t)j] = acc;
        }
    }

    free(E);
    free(F);
}

void solution_free(void) {
    g_ni = g_nj = g_nk = g_nl = g_nm = 0;
    g_A = g_B = g_C = g_D = NULL;
}
