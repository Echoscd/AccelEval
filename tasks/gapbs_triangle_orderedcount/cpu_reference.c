#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

static int g_n = 0;
static const int *g_row_ptr = NULL;
static const int *g_col_idx = NULL;

static void _orbench_old_init(int n, const int *row_ptr, const int *col_idx) {
    g_n = n;
    g_row_ptr = row_ptr;
    g_col_idx = col_idx;
}

static void _orbench_old_compute(unsigned long long *triangle_count_out) {
    uint64_t total = 0;
    if (!triangle_count_out || g_n <= 0 || !g_row_ptr || !g_col_idx) {
        return;
    }

    for (int u = 0; u < g_n; ++u) {
        const int u_begin = g_row_ptr[u];
        const int u_end = g_row_ptr[u + 1];
        for (int e_uv = u_begin; e_uv < u_end; ++e_uv) {
            const int v = g_col_idx[e_uv];
            if (v > u) {
                break;
            }
            int it = g_row_ptr[v];
            const int v_end = g_row_ptr[v + 1];
            for (int e_uw = u_begin; e_uw < u_end; ++e_uw) {
                const int w = g_col_idx[e_uw];
                if (w > v) {
                    break;
                }
                while (it < v_end && g_col_idx[it] < w) {
                    ++it;
                }
                if (it < v_end && g_col_idx[it] == w) {
                    ++total;
                }
            }
        }
    }

    *triangle_count_out = (unsigned long long)total;
}

// ── Unified compute_only wrapper (auto-migrated) ──
void solution_compute(int n, const int * row_ptr, const int * col_idx, unsigned long long * triangle_count_out) {
    _orbench_old_init(n, row_ptr, col_idx);
    _orbench_old_compute(triangle_count_out);
}
