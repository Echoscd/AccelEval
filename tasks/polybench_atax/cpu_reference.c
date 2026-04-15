#include <stdlib.h>

static int g_m = 0;
static int g_n = 0;
static const float *g_A = NULL;
static const float *g_x = NULL;

void solution_init(int m, int n, const float *A, const float *x) {
    g_m = m;
    g_n = n;
    g_A = A;
    g_x = x;
}

void solution_compute(float *y_out) {
    float *tmp = (float*)malloc((size_t)g_m * sizeof(float));
    if (!tmp) return;

    for (int j = 0; j < g_n; ++j) y_out[j] = 0.0f;

    for (int i = 0; i < g_m; ++i) {
        const float *Arow = g_A + (size_t)i * (size_t)g_n;
        float acc = 0.0f;
        for (int j = 0; j < g_n; ++j) {
            acc += Arow[j] * g_x[j];
        }
        tmp[i] = acc;
    }

    for (int i = 0; i < g_m; ++i) {
        const float *Arow = g_A + (size_t)i * (size_t)g_n;
        const float t = tmp[i];
        for (int j = 0; j < g_n; ++j) {
            y_out[j] += Arow[j] * t;
        }
    }

    free(tmp);
}

void solution_free(void) {
    g_m = g_n = 0;
    g_A = NULL;
    g_x = NULL;
}
