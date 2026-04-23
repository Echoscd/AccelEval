#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

typedef struct {
    int n;
    int nnz;
    int max_iters;
    double tol;
    const int *row_ptr;
    const int *col_idx;
    const double *values;
    const double *b;
    double *r;
    double *p;
    double *Ap;
} CGContext;

static CGContext g_ctx = {0};

static double dot_product(int n, const double *a, const double *b) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

static void csr_spmv(int n, const int *row_ptr, const int *col_idx,
                     const double *values, const double *x, double *y) {
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int p = row_ptr[i]; p < row_ptr[i + 1]; ++p) {
            sum += values[p] * x[col_idx[p]];
        }
        y[i] = sum;
    }
}

static void _orbench_old_init(int n, int nnz, int max_iters, int tol_exp,
                   const int *row_ptr, const int *col_idx,
                   const double *values, const double *b) {
    g_ctx.n = n;
    g_ctx.nnz = nnz;
    g_ctx.max_iters = max_iters;
    g_ctx.tol = pow(10.0, -(double)tol_exp);
    g_ctx.row_ptr = row_ptr;
    g_ctx.col_idx = col_idx;
    g_ctx.values = values;
    g_ctx.b = b;
    g_ctx.r = (double*)malloc((size_t)n * sizeof(double));
    g_ctx.p = (double*)malloc((size_t)n * sizeof(double));
    g_ctx.Ap = (double*)malloc((size_t)n * sizeof(double));
}

static void _orbench_old_compute(double *x_out) {
    const int n = g_ctx.n;
    const int *row_ptr = g_ctx.row_ptr;
    const int *col_idx = g_ctx.col_idx;
    const double *values = g_ctx.values;
    const double *b = g_ctx.b;
    double *r = g_ctx.r;
    double *p = g_ctx.p;
    double *Ap = g_ctx.Ap;

    for (int i = 0; i < n; ++i) {
        x_out[i] = 0.0;
        r[i] = b[i];
        p[i] = r[i];
    }

    double rsold = dot_product(n, r, r);
    double tol2 = g_ctx.tol * g_ctx.tol;

    for (int iter = 0; iter < g_ctx.max_iters; ++iter) {
        csr_spmv(n, row_ptr, col_idx, values, p, Ap);
        double denom = dot_product(n, p, Ap);
        if (fabs(denom) < 1e-30) break;
        double alpha = rsold / denom;
        for (int i = 0; i < n; ++i) {
            x_out[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }
        double rsnew = dot_product(n, r, r);
        if (rsnew <= tol2) break;
        double beta = rsnew / rsold;
        for (int i = 0; i < n; ++i) {
            p[i] = r[i] + beta * p[i];
        }
        rsold = rsnew;
    }
}

// ── Unified compute_only wrapper (auto-migrated) ──
void solution_compute(int n, int nnz, int max_iters, int tol_exp, const int * row_ptr, const int * col_idx, const double * values, const double * b, double * x_out) {
    _orbench_old_init(n, nnz, max_iters, tol_exp, row_ptr, col_idx, values, b);
    _orbench_old_compute(x_out);
}
