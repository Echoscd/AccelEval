#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static const float kDamp = 0.85f;
static const float kEps = 1.0e-4f;

static int g_n = 0;
static int g_max_iters = 0;
static const int *g_in_row_ptr = NULL;
static const int *g_in_col_idx = NULL;
static const int *g_out_degree = NULL;
static float *g_scores = NULL;
static float *g_outgoing = NULL;

static void _orbench_old_init(int n,
                   int max_iters,
                   const int *in_row_ptr,
                   const int *in_col_idx,
                   const int *out_degree) {
    g_n = n;
    g_max_iters = max_iters;
    g_in_row_ptr = in_row_ptr;
    g_in_col_idx = in_col_idx;
    g_out_degree = out_degree;
    if (g_scores) free(g_scores);
    if (g_outgoing) free(g_outgoing);
    g_scores = (float*)malloc((size_t)n * sizeof(float));
    g_outgoing = (float*)malloc((size_t)n * sizeof(float));
}

static void _orbench_old_compute(float *scores_out) {
    if (!g_scores || !g_outgoing || g_n <= 0) return;
    const float init_score = 1.0f / (float)g_n;
    const float base_score = (1.0f - kDamp) / (float)g_n;

    for (int i = 0; i < g_n; ++i) {
        g_scores[i] = init_score;
        int od = g_out_degree[i];
        g_outgoing[i] = (od > 0) ? (init_score / (float)od) : 0.0f;
    }

    for (int iter = 0; iter < g_max_iters; ++iter) {
        double error = 0.0;
        for (int u = 0; u < g_n; ++u) {
            float incoming_total = 0.0f;
            for (int e = g_in_row_ptr[u]; e < g_in_row_ptr[u + 1]; ++e) {
                incoming_total += g_outgoing[g_in_col_idx[e]];
            }
            const float old_score = g_scores[u];
            const float new_score = base_score + kDamp * incoming_total;
            g_scores[u] = new_score;
            error += fabs((double)new_score - (double)old_score);
            const int od = g_out_degree[u];
            g_outgoing[u] = (od > 0) ? (new_score / (float)od) : 0.0f;
        }
        if (error < (double)kEps) break;
    }

    memcpy(scores_out, g_scores, (size_t)g_n * sizeof(float));
}

// ── Unified compute_only wrapper (auto-migrated) ──
void solution_compute(int n, int max_iters, const int * in_row_ptr, const int * in_col_idx, const int * out_degree, float * scores_out) {
    _orbench_old_init(n, max_iters, in_row_ptr, in_col_idx, out_degree);
    _orbench_old_compute(scores_out);
}
