#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

typedef struct {
    int n;              // total grid size including boundary
    int interior_n;     // interior size
    double *u;
    double *b;
    double *tmp;
    double *res;
} MGLevel;

static MGLevel *g_levels = NULL;
static int g_num_levels = 0;
static int g_num_cycles = 0;
static int g_pre_smooth = 0;
static int g_post_smooth = 0;
static int g_coarse_iters = 0;

static inline size_t idx3(int n, int x, int y, int z) {
    return ((size_t)z * (size_t)n + (size_t)y) * (size_t)n + (size_t)x;
}

static int build_num_levels(int interior_n) {
    int levels = 1;
    int cur = interior_n;
    while (cur > 4 && (cur % 2) == 0) {
        cur /= 2;
        levels++;
    }
    return levels;
}

static void zero_grid(MGLevel *L, double *a) {
    size_t total = (size_t)L->n * (size_t)L->n * (size_t)L->n;
    memset(a, 0, total * sizeof(double));
}

static void smooth_weighted_jacobi(MGLevel *L, int iters) {
    const int n = L->n;
    const double omega = 2.0 / 3.0;
    for (int it = 0; it < iters; ++it) {
        for (int z = 1; z < n - 1; ++z) {
            for (int y = 1; y < n - 1; ++y) {
                for (int x = 1; x < n - 1; ++x) {
                    size_t c = idx3(n, x, y, z);
                    double sum_nb = L->u[idx3(n, x - 1, y, z)] + L->u[idx3(n, x + 1, y, z)] +
                                    L->u[idx3(n, x, y - 1, z)] + L->u[idx3(n, x, y + 1, z)] +
                                    L->u[idx3(n, x, y, z - 1)] + L->u[idx3(n, x, y, z + 1)];
                    double jac = (L->b[c] + sum_nb) / 6.0;
                    L->tmp[c] = (1.0 - omega) * L->u[c] + omega * jac;
                }
            }
        }
        for (int z = 1; z < n - 1; ++z) {
            for (int y = 1; y < n - 1; ++y) {
                for (int x = 1; x < n - 1; ++x) {
                    size_t c = idx3(n, x, y, z);
                    L->u[c] = L->tmp[c];
                }
            }
        }
    }
}

static void compute_residual(MGLevel *L) {
    const int n = L->n;
    for (int z = 1; z < n - 1; ++z) {
        for (int y = 1; y < n - 1; ++y) {
            for (int x = 1; x < n - 1; ++x) {
                size_t c = idx3(n, x, y, z);
                double Au = 6.0 * L->u[c]
                    - L->u[idx3(n, x - 1, y, z)] - L->u[idx3(n, x + 1, y, z)]
                    - L->u[idx3(n, x, y - 1, z)] - L->u[idx3(n, x, y + 1, z)]
                    - L->u[idx3(n, x, y, z - 1)] - L->u[idx3(n, x, y, z + 1)];
                L->res[c] = L->b[c] - Au;
            }
        }
    }
}

static void restrict_residual(const MGLevel *fine, MGLevel *coarse) {
    zero_grid(coarse, coarse->b);
    const int nc = coarse->n;
    for (int Z = 1; Z < nc - 1; ++Z) {
        for (int Y = 1; Y < nc - 1; ++Y) {
            for (int X = 1; X < nc - 1; ++X) {
                int fx = 2 * X;
                int fy = 2 * Y;
                int fz = 2 * Z;
                double sum = 0.0;
                for (int dz = -1; dz <= 0; ++dz) {
                    for (int dy = -1; dy <= 0; ++dy) {
                        for (int dx = -1; dx <= 0; ++dx) {
                            sum += fine->res[idx3(fine->n, fx + dx, fy + dy, fz + dz)];
                        }
                    }
                }
                coarse->b[idx3(nc, X, Y, Z)] = 0.125 * sum;
            }
        }
    }
}

static void prolongate_and_correct(const MGLevel *coarse, MGLevel *fine) {
    const int nc = coarse->n;
    for (int Z = 1; Z < nc - 1; ++Z) {
        for (int Y = 1; Y < nc - 1; ++Y) {
            for (int X = 1; X < nc - 1; ++X) {
                double corr = coarse->u[idx3(nc, X, Y, Z)];
                int fx = 2 * X;
                int fy = 2 * Y;
                int fz = 2 * Z;
                for (int dz = -1; dz <= 0; ++dz) {
                    for (int dy = -1; dy <= 0; ++dy) {
                        for (int dx = -1; dx <= 0; ++dx) {
                            size_t fi = idx3(fine->n, fx + dx, fy + dy, fz + dz);
                            fine->u[fi] += corr;
                        }
                    }
                }
            }
        }
    }
}

static void vcycle(int level) {
    MGLevel *L = &g_levels[level];
    if (level == g_num_levels - 1) {
        smooth_weighted_jacobi(L, g_coarse_iters);
        return;
    }

    smooth_weighted_jacobi(L, g_pre_smooth);
    compute_residual(L);

    MGLevel *C = &g_levels[level + 1];
    zero_grid(C, C->u);
    restrict_residual(L, C);
    vcycle(level + 1);
    prolongate_and_correct(C, L);
    smooth_weighted_jacobi(L, g_post_smooth);
}

void solution_init(int interior_n, int num_cycles, int pre_smooth, int post_smooth,
                   int coarse_iters, const double *h_rhs) {
    g_num_cycles = num_cycles;
    g_pre_smooth = pre_smooth;
    g_post_smooth = post_smooth;
    g_coarse_iters = coarse_iters;
    g_num_levels = build_num_levels(interior_n);
    g_levels = (MGLevel*)calloc((size_t)g_num_levels, sizeof(MGLevel));

    int cur = interior_n;
    for (int l = 0; l < g_num_levels; ++l) {
        MGLevel *L = &g_levels[l];
        L->interior_n = cur;
        L->n = cur + 2;
        size_t total = (size_t)L->n * (size_t)L->n * (size_t)L->n;
        L->u = (double*)calloc(total, sizeof(double));
        L->b = (double*)calloc(total, sizeof(double));
        L->tmp = (double*)calloc(total, sizeof(double));
        L->res = (double*)calloc(total, sizeof(double));
        if (l == 0) {
            memcpy(L->b, h_rhs, total * sizeof(double));
        }
        if (cur > 4 && (cur % 2) == 0) cur /= 2;
    }
}

void solution_compute(double *h_residual_norm_out) {
    for (int c = 0; c < g_num_cycles; ++c) {
        vcycle(0);
    }
    compute_residual(&g_levels[0]);
    MGLevel *L = &g_levels[0];
    double sumsq = 0.0;
    const int n = L->n;
    for (int z = 1; z < n - 1; ++z) {
        for (int y = 1; y < n - 1; ++y) {
            for (int x = 1; x < n - 1; ++x) {
                double r = L->res[idx3(n, x, y, z)];
                sumsq += r * r;
            }
        }
    }
    h_residual_norm_out[0] = sqrt(sumsq / (double)(L->interior_n * L->interior_n * L->interior_n));
}

void solution_free(void) {
    if (!g_levels) return;
    for (int l = 0; l < g_num_levels; ++l) {
        free(g_levels[l].u);
        free(g_levels[l].b);
        free(g_levels[l].tmp);
        free(g_levels[l].res);
    }
    free(g_levels);
    g_levels = NULL;
    g_num_levels = 0;
}
