#include <stddef.h>

static int g_n = 0;
static const int *g_row_ptr = 0;
static const int *g_col_idx = 0;
static const double *g_values = 0;
static const int *g_diag_idx = 0;
static const double *g_rhs = 0;

static void _orbench_old_init(int n, const int *row_ptr, const int *col_idx,
                   const double *values, const int *diag_idx,
                   const double *rhs) {
    g_n = n;
    g_row_ptr = row_ptr;
    g_col_idx = col_idx;
    g_values = values;
    g_diag_idx = diag_idx;
    g_rhs = rhs;
}

static void _orbench_old_compute(double *x_inout) {
    for (int i = 0; i < g_n; ++i) {
        double sum = g_rhs[i];
        const int row_beg = g_row_ptr[i];
        const int row_end = g_row_ptr[i + 1];
        const int d = g_diag_idx[i];
        const double diag = g_values[d];
        for (int j = row_beg; j < row_end; ++j) {
            sum -= g_values[j] * x_inout[g_col_idx[j]];
        }
        sum += x_inout[i] * diag;
        x_inout[i] = sum / diag;
    }

    for (int i = g_n - 1; i >= 0; --i) {
        double sum = g_rhs[i];
        const int row_beg = g_row_ptr[i];
        const int row_end = g_row_ptr[i + 1];
        const int d = g_diag_idx[i];
        const double diag = g_values[d];
        for (int j = row_beg; j < row_end; ++j) {
            sum -= g_values[j] * x_inout[g_col_idx[j]];
        }
        sum += x_inout[i] * diag;
        x_inout[i] = sum / diag;
    }
}

// ── Unified compute_only wrapper (auto-migrated) ──
void solution_compute(int n, const int * row_ptr, const int * col_idx, const double * values, const int * diag_idx, const double * rhs, double * x_inout) {
    _orbench_old_init(n, row_ptr, col_idx, values, diag_idx, rhs);
    _orbench_old_compute(x_inout);
}
