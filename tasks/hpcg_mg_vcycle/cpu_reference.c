#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct {
    int nx, ny, nz, n;
    int owns_matrix;
    const int *row_ptr;
    const int *col_idx;
    const double *values;
    const int *diag_idx;

    int *row_ptr_owned;
    int *col_idx_owned;
    double *values_owned;
    int *diag_idx_owned;

    int coarse_n;
    int *f2c;
    double *Axf;
    double *rc;
    double *xc;
} Level;

static Level *g_levels = NULL;
static int g_num_levels = 0;
static const double *g_rhs0 = NULL;

static void generate_hpcg_27pt_csr(int nx, int ny, int nz,
                                   int **row_ptr_out,
                                   int **col_idx_out,
                                   double **values_out,
                                   int **diag_idx_out) {
    int n = nx * ny * nz;
    int *row_ptr = (int*)malloc((size_t)(n + 1) * sizeof(int));
    int *col_idx = (int*)malloc((size_t)n * 27u * sizeof(int));
    double *values = (double*)malloc((size_t)n * 27u * sizeof(double));
    int *diag_idx = (int*)malloc((size_t)n * sizeof(int));
    if (!row_ptr || !col_idx || !values || !diag_idx) {
        free(row_ptr); free(col_idx); free(values); free(diag_idx);
        *row_ptr_out = NULL; *col_idx_out = NULL; *values_out = NULL; *diag_idx_out = NULL;
        return;
    }

    int ptr = 0;
    int row = 0;
    int plane = nx * ny;
    row_ptr[0] = 0;
    for (int iz = 0; iz < nz; ++iz) {
        for (int iy = 0; iy < ny; ++iy) {
            int base_row = iz * plane + iy * nx;
            for (int ix = 0; ix < nx; ++ix) {
                int current = base_row + ix;
                int dpos = -1;
                for (int sz = -1; sz <= 1; ++sz) {
                    int z2 = iz + sz;
                    if (z2 < 0 || z2 >= nz) continue;
                    int zoff = z2 * plane;
                    for (int sy = -1; sy <= 1; ++sy) {
                        int y2 = iy + sy;
                        if (y2 < 0 || y2 >= ny) continue;
                        int yoff = zoff + y2 * nx;
                        for (int sx = -1; sx <= 1; ++sx) {
                            int x2 = ix + sx;
                            if (x2 < 0 || x2 >= nx) continue;
                            int col = yoff + x2;
                            col_idx[ptr] = col;
                            if (col == current) {
                                values[ptr] = 26.0;
                                dpos = ptr;
                            } else {
                                values[ptr] = -1.0;
                            }
                            ++ptr;
                        }
                    }
                }
                diag_idx[row] = dpos;
                ++row;
                row_ptr[row] = ptr;
            }
        }
    }

    *row_ptr_out = row_ptr;
    *col_idx_out = (int*)realloc(col_idx, (size_t)ptr * sizeof(int));
    if (!*col_idx_out) *col_idx_out = col_idx;
    *values_out = (double*)realloc(values, (size_t)ptr * sizeof(double));
    if (!*values_out) *values_out = values;
    *diag_idx_out = diag_idx;
}

static void symgs(const Level *L, const double *rhs, double *x) {
    for (int i = 0; i < L->n; ++i) {
        double sum = rhs[i];
        int row_beg = L->row_ptr[i];
        int row_end = L->row_ptr[i + 1];
        int d = L->diag_idx[i];
        double diag = L->values[d];
        for (int j = row_beg; j < row_end; ++j) sum -= L->values[j] * x[L->col_idx[j]];
        sum += x[i] * diag;
        x[i] = sum / diag;
    }
    for (int i = L->n - 1; i >= 0; --i) {
        double sum = rhs[i];
        int row_beg = L->row_ptr[i];
        int row_end = L->row_ptr[i + 1];
        int d = L->diag_idx[i];
        double diag = L->values[d];
        for (int j = row_beg; j < row_end; ++j) sum -= L->values[j] * x[L->col_idx[j]];
        sum += x[i] * diag;
        x[i] = sum / diag;
    }
}

static void spmv(const Level *L, const double *x, double *y) {
    for (int i = 0; i < L->n; ++i) {
        double sum = 0.0;
        for (int j = L->row_ptr[i]; j < L->row_ptr[i + 1]; ++j) {
            sum += L->values[j] * x[L->col_idx[j]];
        }
        y[i] = sum;
    }
}

static void mg_vcycle(int level_idx, const double *rhs, double *x) {
    Level *L = &g_levels[level_idx];
    if (level_idx == g_num_levels - 1) {
        symgs(L, rhs, x);
        return;
    }

    symgs(L, rhs, x);
    spmv(L, x, L->Axf);
    for (int i = 0; i < L->coarse_n; ++i) {
        int fi = L->f2c[i];
        L->rc[i] = rhs[fi] - L->Axf[fi];
        L->xc[i] = 0.0;
    }

    mg_vcycle(level_idx + 1, L->rc, L->xc);

    for (int i = 0; i < L->coarse_n; ++i) {
        x[L->f2c[i]] += L->xc[i];
    }

    symgs(L, rhs, x);
}

static void _orbench_old_init(int nx, int ny, int nz, int coarse_levels,
                   int n, const int *row_ptr, const int *col_idx,
                   const double *values, const int *diag_idx,
                   const double *rhs) {
    g_num_levels = coarse_levels + 1;
    g_levels = (Level*)calloc((size_t)g_num_levels, sizeof(Level));
    if (!g_levels) return;

    g_levels[0].nx = nx; g_levels[0].ny = ny; g_levels[0].nz = nz; g_levels[0].n = n;
    g_levels[0].owns_matrix = 0;
    g_levels[0].row_ptr = row_ptr;
    g_levels[0].col_idx = col_idx;
    g_levels[0].values = values;
    g_levels[0].diag_idx = diag_idx;
    g_rhs0 = rhs;

    for (int l = 0; l < g_num_levels - 1; ++l) {
        Level *fine = &g_levels[l];
        int cnx = fine->nx / 2;
        int cny = fine->ny / 2;
        int cnz = fine->nz / 2;
        int cn = cnx * cny * cnz;
        fine->coarse_n = cn;
        fine->f2c = (int*)malloc((size_t)cn * sizeof(int));
        fine->Axf = (double*)malloc((size_t)fine->n * sizeof(double));
        fine->rc = (double*)malloc((size_t)cn * sizeof(double));
        fine->xc = (double*)malloc((size_t)cn * sizeof(double));

        int idx = 0;
        int fine_plane = fine->nx * fine->ny;
        for (int iz = 0; iz < cnz; ++iz) {
            int fiz = 2 * iz;
            int zoff = fiz * fine_plane;
            for (int iy = 0; iy < cny; ++iy) {
                int fiy = 2 * iy;
                int yoff = zoff + fiy * fine->nx;
                for (int ix = 0; ix < cnx; ++ix) {
                    fine->f2c[idx++] = yoff + 2 * ix;
                }
            }
        }

        Level *coarse = &g_levels[l + 1];
        coarse->nx = cnx; coarse->ny = cny; coarse->nz = cnz; coarse->n = cn;
        coarse->owns_matrix = 1;
        generate_hpcg_27pt_csr(cnx, cny, cnz,
                               &coarse->row_ptr_owned,
                               &coarse->col_idx_owned,
                               &coarse->values_owned,
                               &coarse->diag_idx_owned);
        coarse->row_ptr = coarse->row_ptr_owned;
        coarse->col_idx = coarse->col_idx_owned;
        coarse->values = coarse->values_owned;
        coarse->diag_idx = coarse->diag_idx_owned;
    }
}

static void _orbench_old_compute(double *x_inout) {
    if (!g_levels) return;
    mg_vcycle(0, g_rhs0, x_inout);
}

// ── Unified compute_only wrapper (auto-migrated) ──
void solution_compute(int nx, int ny, int nz, int coarse_levels, int n, const int * row_ptr, const int * col_idx, const double * values, const int * diag_idx, const double * rhs, double * x_inout) {
    _orbench_old_init(nx, ny, nz, coarse_levels, n, row_ptr, col_idx, values, diag_idx, rhs);
    _orbench_old_compute(x_inout);
}
