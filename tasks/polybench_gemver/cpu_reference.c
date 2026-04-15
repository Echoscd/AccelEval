#include <stdlib.h>

static int g_n = 0;
static const float *g_A = NULL;
static const float *g_u1 = NULL;
static const float *g_v1 = NULL;
static const float *g_u2 = NULL;
static const float *g_v2 = NULL;
static const float *g_y = NULL;
static const float *g_z = NULL;

void solution_init(int n,
                   const float *A,
                   const float *u1,
                   const float *v1,
                   const float *u2,
                   const float *v2,
                   const float *y,
                   const float *z) {
    g_n = n;
    g_A = A;
    g_u1 = u1;
    g_v1 = v1;
    g_u2 = u2;
    g_v2 = v2;
    g_y = y;
    g_z = z;
}

void solution_compute(float *w_out) {
    const float alpha = 1.5f;
    const float beta = 1.2f;
    const int n = g_n;

    float *Aupd = (float*)malloc((size_t)n * (size_t)n * sizeof(float));
    float *x = (float*)malloc((size_t)n * sizeof(float));
    if (!Aupd || !x) {
        free(Aupd);
        free(x);
        return;
    }

    for (int i = 0; i < n; ++i) {
        const float ui1 = g_u1[i];
        const float ui2 = g_u2[i];
        const float *Arow = g_A + (size_t)i * (size_t)n;
        float *Urow = Aupd + (size_t)i * (size_t)n;
        for (int j = 0; j < n; ++j) {
            Urow[j] = Arow[j] + ui1 * g_v1[j] + ui2 * g_v2[j];
        }
    }

    for (int i = 0; i < n; ++i) {
        float acc = 0.0f;
        for (int j = 0; j < n; ++j) {
            acc += beta * Aupd[(size_t)j * (size_t)n + (size_t)i] * g_y[j];
        }
        x[i] = acc + g_z[i];
    }

    for (int i = 0; i < n; ++i) {
        const float *Urow = Aupd + (size_t)i * (size_t)n;
        float acc = 0.0f;
        for (int j = 0; j < n; ++j) {
            acc += alpha * Urow[j] * x[j];
        }
        w_out[i] = acc;
    }

    free(Aupd);
    free(x);
}

void solution_free(void) {
    g_n = 0;
    g_A = NULL;
    g_u1 = NULL;
    g_v1 = NULL;
    g_u2 = NULL;
    g_v2 = NULL;
    g_y = NULL;
    g_z = NULL;
}
