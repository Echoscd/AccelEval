#include <stdlib.h>
#include <string.h>

static int g_n = 0, g_m = 0;
static const float *g_A = NULL;
static const float *g_p = NULL;
static const float *g_r = NULL;

void solution_init(int n, int m, const float *A, const float *p, const float *r) {
    g_n = n;
    g_m = m;
    g_A = A;
    g_p = p;
    g_r = r;
}

void solution_compute(float *s_out, float *q_out) {
    for (int j = 0; j < g_m; ++j) s_out[j] = 0.0f;

    for (int i = 0; i < g_n; ++i) {
        const float *Arow = g_A + (size_t)i * (size_t)g_m;
        float qi = 0.0f;
        float ri = g_r[i];
        for (int j = 0; j < g_m; ++j) {
            float aij = Arow[j];
            s_out[j] += ri * aij;
            qi += aij * g_p[j];
        }
        q_out[i] = qi;
    }
}

void solution_free(void) {
    g_n = g_m = 0;
    g_A = NULL;
    g_p = NULL;
    g_r = NULL;
}
