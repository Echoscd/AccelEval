#include <stdlib.h>
#include <string.h>

static int g_n = 0;
static int g_tsteps = 0;
static const float *g_A0 = NULL;
static const float *g_B0 = NULL;

void solution_init(int n, int tsteps, const float *A0, const float *B0) {
    g_n = n;
    g_tsteps = tsteps;
    g_A0 = A0;
    g_B0 = B0;
}

void solution_compute(float *A_out) {
    const size_t total = (size_t)g_n * (size_t)g_n;
    float *A = (float*)malloc(total * sizeof(float));
    float *B = (float*)malloc(total * sizeof(float));
    if (!A || !B) {
        free(A);
        free(B);
        return;
    }

    memcpy(A, g_A0, total * sizeof(float));
    memcpy(B, g_B0, total * sizeof(float));

    for (int t = 0; t < g_tsteps; ++t) {
        for (int i = 1; i < g_n - 1; ++i) {
            const size_t row = (size_t)i * (size_t)g_n;
            const size_t row_up = (size_t)(i - 1) * (size_t)g_n;
            const size_t row_dn = (size_t)(i + 1) * (size_t)g_n;
            for (int j = 1; j < g_n - 1; ++j) {
                B[row + (size_t)j] = 0.2f * (
                    A[row + (size_t)j] +
                    A[row + (size_t)(j - 1)] +
                    A[row + (size_t)(j + 1)] +
                    A[row_up + (size_t)j] +
                    A[row_dn + (size_t)j]
                );
            }
        }
        for (int i = 1; i < g_n - 1; ++i) {
            const size_t row = (size_t)i * (size_t)g_n;
            const size_t row_up = (size_t)(i - 1) * (size_t)g_n;
            const size_t row_dn = (size_t)(i + 1) * (size_t)g_n;
            for (int j = 1; j < g_n - 1; ++j) {
                A[row + (size_t)j] = 0.2f * (
                    B[row + (size_t)j] +
                    B[row + (size_t)(j - 1)] +
                    B[row + (size_t)(j + 1)] +
                    B[row_up + (size_t)j] +
                    B[row_dn + (size_t)j]
                );
            }
        }
    }

    memcpy(A_out, A, total * sizeof(float));
    free(A);
    free(B);
}

void solution_free(void) {
    g_n = 0;
    g_tsteps = 0;
    g_A0 = NULL;
    g_B0 = NULL;
}
