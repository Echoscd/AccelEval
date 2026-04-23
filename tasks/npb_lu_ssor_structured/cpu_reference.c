#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_n = 0;
static int g_iters = 0;
static double g_omega = 1.1;
static double *g_u0 = NULL;
static double *g_rhs = NULL;
static size_t g_count = 0;

#define NVARS 5

static inline size_t idx4(int n, int i, int j, int k, int m) {
    return (((size_t)k * (size_t)n + (size_t)j) * (size_t)n + (size_t)i) * NVARS + (size_t)m;
}

static void copy5(double dst[5], const double *src) {
    for (int m = 0; m < NVARS; ++m) dst[m] = src[m];
}

static void solve_5x5(const double A[25], const double b[5], double x[5]) {
    double M[25];
    double rhs[5];
    for (int i = 0; i < 25; ++i) M[i] = A[i];
    for (int i = 0; i < 5; ++i) rhs[i] = b[i];

    for (int col = 0; col < 5; ++col) {
        int pivot = col;
        double best = fabs(M[col * 5 + col]);
        for (int r = col + 1; r < 5; ++r) {
            double v = fabs(M[r * 5 + col]);
            if (v > best) { best = v; pivot = r; }
        }
        if (pivot != col) {
            for (int c = col; c < 5; ++c) {
                double tmp = M[col * 5 + c];
                M[col * 5 + c] = M[pivot * 5 + c];
                M[pivot * 5 + c] = tmp;
            }
            double tr = rhs[col]; rhs[col] = rhs[pivot]; rhs[pivot] = tr;
        }
        double diag = M[col * 5 + col];
        if (fabs(diag) < 1e-12) diag = (diag >= 0.0 ? 1e-12 : -1e-12);
        for (int r = col + 1; r < 5; ++r) {
            double factor = M[r * 5 + col] / diag;
            M[r * 5 + col] = 0.0;
            for (int c = col + 1; c < 5; ++c) M[r * 5 + c] -= factor * M[col * 5 + c];
            rhs[r] -= factor * rhs[col];
        }
    }
    for (int r = 4; r >= 0; --r) {
        double sum = rhs[r];
        for (int c = r + 1; c < 5; ++c) sum -= M[r * 5 + c] * x[c];
        double diag = M[r * 5 + r];
        if (fabs(diag) < 1e-12) diag = (diag >= 0.0 ? 1e-12 : -1e-12);
        x[r] = sum / diag;
    }
}

static void build_local_block(double B[25]) {
    static const double diag[NVARS] = {4.40, 4.85, 5.20, 5.55, 5.90};
    for (int r = 0; r < NVARS; ++r) {
        for (int c = 0; c < NVARS; ++c) {
            if (r == c) B[r * 5 + c] = diag[r];
            else B[r * 5 + c] = 0.0125 * (double)(r + 1) * (double)(c + 1);
        }
    }
}

static void point_update(double *u, int i, int j, int k) {
    static const double cx = 0.14;
    static const double cy = 0.11;
    static const double cz = 0.09;
    double B[25];
    double rhs_loc[5];
    double sol[5] = {0,0,0,0,0};
    build_local_block(B);

    for (int m = 0; m < NVARS; ++m) {
        size_t id = idx4(g_n, i, j, k, m);
        double s = g_rhs[id];
        if (i > 0)       s += cx * u[idx4(g_n, i - 1, j, k, m)];
        if (i + 1 < g_n) s += cx * u[idx4(g_n, i + 1, j, k, m)];
        if (j > 0)       s += cy * u[idx4(g_n, i, j - 1, k, m)];
        if (j + 1 < g_n) s += cy * u[idx4(g_n, i, j + 1, k, m)];
        if (k > 0)       s += cz * u[idx4(g_n, i, j, k - 1, m)];
        if (k + 1 < g_n) s += cz * u[idx4(g_n, i, j, k + 1, m)];
        rhs_loc[m] = s;
    }

    solve_5x5(B, rhs_loc, sol);
    for (int m = 0; m < NVARS; ++m) {
        size_t id = idx4(g_n, i, j, k, m);
        u[id] = (1.0 - g_omega) * u[id] + g_omega * sol[m];
    }
}

static void compute_residual_norms(const double *u, double out[5]) {
    static const double cx = 0.14;
    static const double cy = 0.11;
    static const double cz = 0.09;
    double B[25];
    build_local_block(B);
    double sums[5] = {0,0,0,0,0};
    const double denom = (double)((size_t)g_n * (size_t)g_n * (size_t)g_n);

    for (int k = 0; k < g_n; ++k) {
        for (int j = 0; j < g_n; ++j) {
            for (int i = 0; i < g_n; ++i) {
                double Au[5] = {0,0,0,0,0};
                for (int r = 0; r < NVARS; ++r) {
                    for (int c = 0; c < NVARS; ++c) {
                        Au[r] += B[r * 5 + c] * u[idx4(g_n, i, j, k, c)];
                    }
                    if (i > 0)       Au[r] -= cx * u[idx4(g_n, i - 1, j, k, r)];
                    if (i + 1 < g_n) Au[r] -= cx * u[idx4(g_n, i + 1, j, k, r)];
                    if (j > 0)       Au[r] -= cy * u[idx4(g_n, i, j - 1, k, r)];
                    if (j + 1 < g_n) Au[r] -= cy * u[idx4(g_n, i, j + 1, k, r)];
                    if (k > 0)       Au[r] -= cz * u[idx4(g_n, i, j, k - 1, r)];
                    if (k + 1 < g_n) Au[r] -= cz * u[idx4(g_n, i, j, k + 1, r)];
                }
                for (int m = 0; m < NVARS; ++m) {
                    double res = g_rhs[idx4(g_n, i, j, k, m)] - Au[m];
                    sums[m] += res * res;
                }
            }
        }
    }

    for (int m = 0; m < NVARS; ++m) out[m] = sqrt(sums[m] / denom);
}

static void _orbench_old_init(int n, int iters, int omega_milli, const double *u0, const double *rhs) {
    g_n = n;
    g_iters = iters;
    g_omega = ((double)omega_milli) / 1000.0;
    g_count = (size_t)n * (size_t)n * (size_t)n * NVARS;

    free(g_u0); g_u0 = NULL;
    free(g_rhs); g_rhs = NULL;
    g_u0 = (double*)malloc(g_count * sizeof(double));
    g_rhs = (double*)malloc(g_count * sizeof(double));
    if (!g_u0 || !g_rhs) {
        fprintf(stderr, "[cpu_reference] OOM in solution_init\n");
        exit(1);
    }
    memcpy(g_u0, u0, g_count * sizeof(double));
    memcpy(g_rhs, rhs, g_count * sizeof(double));
}

static void _orbench_old_compute(double *residual_out) {
    double *u = (double*)malloc(g_count * sizeof(double));
    if (!u) {
        fprintf(stderr, "[cpu_reference] OOM in solution_compute\n");
        exit(1);
    }
    memcpy(u, g_u0, g_count * sizeof(double));

    for (int it = 0; it < g_iters; ++it) {
        for (int k = 0; k < g_n; ++k)
            for (int j = 0; j < g_n; ++j)
                for (int i = 0; i < g_n; ++i)
                    point_update(u, i, j, k);

        for (int k = g_n - 1; k >= 0; --k)
            for (int j = g_n - 1; j >= 0; --j)
                for (int i = g_n - 1; i >= 0; --i)
                    point_update(u, i, j, k);
    }

    compute_residual_norms(u, residual_out);
    free(u);
}

// ── Unified compute_only wrapper (auto-migrated) ──
void solution_compute(int n, int iters, int omega_milli, const double * u0, const double * rhs, double * residual_out) {
    _orbench_old_init(n, iters, omega_milli, u0, rhs);
    _orbench_old_compute(residual_out);
}
