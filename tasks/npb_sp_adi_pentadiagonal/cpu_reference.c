#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_n = 0;
static int g_iters = 0;
static double g_omega = 1.0;
static double *g_u = NULL;
static double *g_rhs = NULL;
static double *g_res = NULL;
static double *g_tmp1 = NULL;
static double *g_tmp2 = NULL;
static double *g_tmp3 = NULL;
static double *g_aux = NULL;

static inline size_t idx4(int k, int j, int i, int m) {
    return ((((size_t)k * (size_t)g_n + (size_t)j) * (size_t)g_n + (size_t)i) * 5u + (size_t)m);
}

static inline double axis_a2(int axis, int m) {
    static const double base[3] = {0.010, 0.012, 0.015};
    return base[axis] * (1.0 + 0.04 * (double)m);
}

static inline double axis_a1(int axis, int m) {
    static const double base[3] = {0.065, 0.075, 0.085};
    return base[axis] * (1.0 + 0.03 * (double)m);
}

static inline double axis_center(int axis, int m) {
    double a2 = axis_a2(axis, m);
    double a1 = axis_a1(axis, m);
    return 1.0 + 2.0 * a1 + 2.0 * a2 + 0.01 * axis + 0.005 * m;
}

static void apply_operator(const double *u, double *out) {
    for (int k = 0; k < g_n; ++k) {
        for (int j = 0; j < g_n; ++j) {
            for (int i = 0; i < g_n; ++i) {
                for (int m = 0; m < 5; ++m) {
                    size_t id = idx4(k, j, i, m);
                    double center = 1.0;
                    double acc = center * u[id];

                    for (int axis = 0; axis < 3; ++axis) {
                        double a1 = axis_a1(axis, m);
                        double a2 = axis_a2(axis, m);
                        int i1, i2, i3, i4;
                        if (axis == 0) {
                            i1 = i - 1; i2 = i + 1; i3 = i - 2; i4 = i + 2;
                            acc += 2.0 * (a1 + a2) * u[id];
                            if (i1 >= 0) acc -= a1 * u[idx4(k, j, i1, m)];
                            if (i2 < g_n) acc -= a1 * u[idx4(k, j, i2, m)];
                            if (i3 >= 0) acc -= a2 * u[idx4(k, j, i3, m)];
                            if (i4 < g_n) acc -= a2 * u[idx4(k, j, i4, m)];
                        } else if (axis == 1) {
                            i1 = j - 1; i2 = j + 1; i3 = j - 2; i4 = j + 2;
                            acc += 2.0 * (a1 + a2) * u[id];
                            if (i1 >= 0) acc -= a1 * u[idx4(k, i1, i, m)];
                            if (i2 < g_n) acc -= a1 * u[idx4(k, i2, i, m)];
                            if (i3 >= 0) acc -= a2 * u[idx4(k, i3, i, m)];
                            if (i4 < g_n) acc -= a2 * u[idx4(k, i4, i, m)];
                        } else {
                            i1 = k - 1; i2 = k + 1; i3 = k - 2; i4 = k + 2;
                            acc += 2.0 * (a1 + a2) * u[id];
                            if (i1 >= 0) acc -= a1 * u[idx4(i1, j, i, m)];
                            if (i2 < g_n) acc -= a1 * u[idx4(i2, j, i, m)];
                            if (i3 >= 0) acc -= a2 * u[idx4(i3, j, i, m)];
                            if (i4 < g_n) acc -= a2 * u[idx4(i4, j, i, m)];
                        }
                    }
                    out[id] = acc;
                }
            }
        }
    }
}

static void txinvr_like(double *buf) {
    size_t total = (size_t)g_n * (size_t)g_n * (size_t)g_n;
    for (size_t p = 0; p < total; ++p) {
        double v0 = buf[p * 5 + 0];
        double v1 = buf[p * 5 + 1];
        double v2 = buf[p * 5 + 2];
        double v3 = buf[p * 5 + 3];
        double v4 = buf[p * 5 + 4];
        buf[p * 5 + 0] = 0.98 * v0 + 0.02 * v1;
        buf[p * 5 + 1] = -0.04 * v0 + 1.01 * v1 + 0.03 * v2;
        buf[p * 5 + 2] = 0.02 * v0 - 0.03 * v1 + 1.00 * v2 + 0.02 * v3;
        buf[p * 5 + 3] = 0.01 * v1 + 0.04 * v2 + 0.99 * v3 + 0.02 * v4;
        buf[p * 5 + 4] = -0.02 * v2 + 0.03 * v3 + 1.00 * v4;
    }
}

static void ninvr_like(double *buf) {
    size_t total = (size_t)g_n * (size_t)g_n * (size_t)g_n;
    for (size_t p = 0; p < total; ++p) {
        double v0 = buf[p * 5 + 0];
        double v1 = buf[p * 5 + 1];
        double v2 = buf[p * 5 + 2];
        double v3 = buf[p * 5 + 3];
        double v4 = buf[p * 5 + 4];
        buf[p * 5 + 0] = 1.00 * v0 + 0.01 * (v3 - v4);
        buf[p * 5 + 1] = 0.99 * v1 + 0.02 * v0;
        buf[p * 5 + 2] = 1.01 * v2 - 0.01 * v1 + 0.01 * v4;
        buf[p * 5 + 3] = 0.98 * v3 + 0.03 * v2;
        buf[p * 5 + 4] = 1.00 * v4 + 0.02 * v3 - 0.01 * v0;
    }
}

static void tzetar_like(double *buf) {
    size_t total = (size_t)g_n * (size_t)g_n * (size_t)g_n;
    for (size_t p = 0; p < total; ++p) {
        double v0 = buf[p * 5 + 0];
        double v1 = buf[p * 5 + 1];
        double v2 = buf[p * 5 + 2];
        double v3 = buf[p * 5 + 3];
        double v4 = buf[p * 5 + 4];
        buf[p * 5 + 0] = v0 + 0.015 * v4;
        buf[p * 5 + 1] = v1 + 0.010 * v3;
        buf[p * 5 + 2] = v2 + 0.012 * (v0 + v4);
        buf[p * 5 + 3] = 0.985 * v3 + 0.020 * v2;
        buf[p * 5 + 4] = 0.990 * v4 + 0.015 * v1;
    }
}

static void pinvr_like(double *buf) {
    size_t total = (size_t)g_n * (size_t)g_n * (size_t)g_n;
    for (size_t p = 0; p < total; ++p) {
        double v0 = buf[p * 5 + 0];
        double v1 = buf[p * 5 + 1];
        double v2 = buf[p * 5 + 2];
        double v3 = buf[p * 5 + 3];
        double v4 = buf[p * 5 + 4];
        buf[p * 5 + 0] = 0.995 * v0 + 0.010 * v2;
        buf[p * 5 + 1] = 1.000 * v1 - 0.012 * v4;
        buf[p * 5 + 2] = 0.990 * v2 + 0.010 * (v1 + v3);
        buf[p * 5 + 3] = 1.005 * v3 - 0.010 * v0;
        buf[p * 5 + 4] = 0.995 * v4 + 0.008 * v2;
    }
}

static void solve_pentadiagonal_line(int axis, int m, const double *rhs_line, double *x_line) {
    int n = g_n;
    double a[n], b[n], c[n], d[n], e[n], f[n], alpha[n], beta[n], gamma[n];
    double aa2 = axis_a2(axis, m);
    double aa1 = axis_a1(axis, m);
    double cc = axis_center(axis, m);

    for (int t = 0; t < n; ++t) {
        a[t] = 0.0;
        b[t] = 0.0;
        c[t] = 1.0;
        d[t] = 0.0;
        e[t] = 0.0;
        f[t] = rhs_line[t];
    }

    for (int t = 2; t <= n - 3; ++t) {
        a[t] = -aa2;
        b[t] = -aa1;
        c[t] = cc;
        d[t] = -aa1;
        e[t] = -aa2;
    }

    double denom = c[0];
    alpha[0] = (n > 1) ? d[0] / denom : 0.0;
    beta[0] = (n > 2) ? e[0] / denom : 0.0;
    gamma[0] = f[0] / denom;

    if (n > 1) {
        denom = c[1] - b[1] * alpha[0];
        alpha[1] = (n > 2) ? (d[1] - b[1] * beta[0]) / denom : 0.0;
        beta[1] = (n > 3) ? e[1] / denom : 0.0;
        gamma[1] = (f[1] - b[1] * gamma[0]) / denom;
    }

    for (int t = 2; t < n; ++t) {
        denom = c[t] - a[t] * beta[t - 2] - b[t] * alpha[t - 1];
        alpha[t] = (t <= n - 2) ? (d[t] - b[t] * beta[t - 1]) / denom : 0.0;
        beta[t] = (t <= n - 3) ? e[t] / denom : 0.0;
        gamma[t] = (f[t] - a[t] * gamma[t - 2] - b[t] * gamma[t - 1]) / denom;
    }

    x_line[n - 1] = gamma[n - 1];
    if (n > 1) x_line[n - 2] = gamma[n - 2] - alpha[n - 2] * x_line[n - 1];
    for (int t = n - 3; t >= 0; --t) {
        x_line[t] = gamma[t] - alpha[t] * x_line[t + 1] - beta[t] * x_line[t + 2];
    }
}

static void solve_axis(int axis, const double *in, double *out) {
    int n = g_n;
    double rhs_line[n], x_line[n];

    if (axis == 0) {
        for (int k = 0; k < n; ++k) {
            for (int j = 0; j < n; ++j) {
                for (int m = 0; m < 5; ++m) {
                    for (int i = 0; i < n; ++i) rhs_line[i] = in[idx4(k, j, i, m)];
                    solve_pentadiagonal_line(axis, m, rhs_line, x_line);
                    for (int i = 0; i < n; ++i) out[idx4(k, j, i, m)] = x_line[i];
                }
            }
        }
    } else if (axis == 1) {
        for (int k = 0; k < n; ++k) {
            for (int i = 0; i < n; ++i) {
                for (int m = 0; m < 5; ++m) {
                    for (int j = 0; j < n; ++j) rhs_line[j] = in[idx4(k, j, i, m)];
                    solve_pentadiagonal_line(axis, m, rhs_line, x_line);
                    for (int j = 0; j < n; ++j) out[idx4(k, j, i, m)] = x_line[j];
                }
            }
        }
    } else {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                for (int m = 0; m < 5; ++m) {
                    for (int k = 0; k < n; ++k) rhs_line[k] = in[idx4(k, j, i, m)];
                    solve_pentadiagonal_line(axis, m, rhs_line, x_line);
                    for (int k = 0; k < n; ++k) out[idx4(k, j, i, m)] = x_line[k];
                }
            }
        }
    }
}

void solution_init(int n, int iters, int omega_milli, const double *u0, const double *rhs) {
    g_n = n;
    g_iters = iters;
    g_omega = ((double)omega_milli) / 1000.0;
    size_t total = (size_t)n * (size_t)n * (size_t)n * 5u;

    g_u = (double*)malloc(total * sizeof(double));
    g_rhs = (double*)malloc(total * sizeof(double));
    g_res = (double*)malloc(total * sizeof(double));
    g_tmp1 = (double*)malloc(total * sizeof(double));
    g_tmp2 = (double*)malloc(total * sizeof(double));
    g_tmp3 = (double*)malloc(total * sizeof(double));
    g_aux = (double*)malloc(total * sizeof(double));
    if (!g_u || !g_rhs || !g_res || !g_tmp1 || !g_tmp2 || !g_tmp3 || !g_aux) {
        fprintf(stderr, "[npb_sp] allocation failed\n");
        exit(1);
    }
    memcpy(g_u, u0, total * sizeof(double));
    memcpy(g_rhs, rhs, total * sizeof(double));
}

void solution_compute(double *residual_out) {
    size_t total = (size_t)g_n * (size_t)g_n * (size_t)g_n * 5u;
    for (int it = 0; it < g_iters; ++it) {
        apply_operator(g_u, g_aux);
        for (size_t p = 0; p < total; ++p) g_res[p] = g_rhs[p] - g_aux[p];
        txinvr_like(g_res);
        solve_axis(0, g_res, g_tmp1);
        ninvr_like(g_tmp1);
        solve_axis(1, g_tmp1, g_tmp2);
        tzetar_like(g_tmp2);
        solve_axis(2, g_tmp2, g_tmp3);
        pinvr_like(g_tmp3);
        for (size_t p = 0; p < total; ++p) g_u[p] += g_omega * g_tmp3[p];
    }

    apply_operator(g_u, g_aux);
    for (int m = 0; m < 5; ++m) residual_out[m] = 0.0;
    size_t cells = (size_t)g_n * (size_t)g_n * (size_t)g_n;
    for (size_t p = 0; p < cells; ++p) {
        for (int m = 0; m < 5; ++m) {
            double r = g_rhs[p * 5u + (size_t)m] - g_aux[p * 5u + (size_t)m];
            residual_out[m] += r * r;
        }
    }
    for (int m = 0; m < 5; ++m) residual_out[m] = sqrt(residual_out[m] / (double)cells);
}

void solution_free(void) {
    free(g_u); free(g_rhs); free(g_res); free(g_tmp1); free(g_tmp2); free(g_tmp3); free(g_aux);
    g_u = g_rhs = g_res = g_tmp1 = g_tmp2 = g_tmp3 = g_aux = NULL;
    g_n = 0;
    g_iters = 0;
    g_omega = 1.0;
}
