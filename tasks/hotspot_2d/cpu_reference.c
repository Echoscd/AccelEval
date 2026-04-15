// Adapted from the Rodinia Hotspot benchmark core solver
// Source repo: https://github.com/yuhc/gpu-rodinia
// Original file: openmp/hotspot/hotspot_openmp.cpp

#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef float FLOAT;

#define MAX_PD (3.0e6f)
#define PRECISION 0.001f
#define SPEC_HEAT_SI 1.75e6f
#define K_SI 100.0f
#define FACTOR_CHIP 0.5f

static const FLOAT t_chip = 0.0005f;
static const FLOAT chip_height = 0.016f;
static const FLOAT chip_width = 0.016f;
static const FLOAT amb_temp = 80.0f;

static int g_rows = 0;
static int g_cols = 0;
static int g_iters = 0;
static const FLOAT* g_temp0 = NULL;
static const FLOAT* g_power = NULL;
static FLOAT* g_buf_a = NULL;
static FLOAT* g_buf_b = NULL;

void solution_init(int rows, int cols, int iters,
                   const float* temp0,
                   const float* power) {
    g_rows = rows;
    g_cols = cols;
    g_iters = iters;
    g_temp0 = temp0;
    g_power = power;

    size_t n = (size_t)rows * (size_t)cols;
    if (g_buf_a) free(g_buf_a);
    if (g_buf_b) free(g_buf_b);
    g_buf_a = (FLOAT*)malloc(n * sizeof(FLOAT));
    g_buf_b = (FLOAT*)malloc(n * sizeof(FLOAT));
}

void solution_compute(float* out_temp) {
    if (!g_temp0 || !g_power || !g_buf_a || !g_buf_b || !out_temp ||
        g_rows <= 0 || g_cols <= 0 || g_iters <= 0) {
        return;
    }

    int row = g_rows;
    int col = g_cols;
    size_t n = (size_t)row * (size_t)col;
    memcpy(g_buf_a, g_temp0, n * sizeof(FLOAT));

    FLOAT grid_height = chip_height / (FLOAT)row;
    FLOAT grid_width  = chip_width  / (FLOAT)col;
    FLOAT Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    FLOAT Rx  = grid_width  / (2.0f * K_SI * t_chip * grid_height);
    FLOAT Ry  = grid_height / (2.0f * K_SI * t_chip * grid_width);
    FLOAT Rz  = t_chip / (K_SI * grid_height * grid_width);
    FLOAT max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    FLOAT step = PRECISION / max_slope / 1000.0f;
    FLOAT Rx_1 = 1.0f / Rx;
    FLOAT Ry_1 = 1.0f / Ry;
    FLOAT Rz_1 = 1.0f / Rz;
    FLOAT Cap_1 = step / Cap;

    FLOAT* src = g_buf_a;
    FLOAT* dst = g_buf_b;

    for (int iter = 0; iter < g_iters; ++iter) {
        for (int r = 0; r < row; ++r) {
            for (int c = 0; c < col; ++c) {
                int idx = r * col + c;
                FLOAT delta;

                if (r == 0 && c == 0) {
                    delta = Cap_1 * (g_power[idx]
                        + (src[idx + 1] - src[idx]) * Rx_1
                        + (src[idx + col] - src[idx]) * Ry_1
                        + (amb_temp - src[idx]) * Rz_1);
                } else if (r == 0 && c == col - 1) {
                    delta = Cap_1 * (g_power[idx]
                        + (src[idx - 1] - src[idx]) * Rx_1
                        + (src[idx + col] - src[idx]) * Ry_1
                        + (amb_temp - src[idx]) * Rz_1);
                } else if (r == row - 1 && c == col - 1) {
                    delta = Cap_1 * (g_power[idx]
                        + (src[idx - 1] - src[idx]) * Rx_1
                        + (src[idx - col] - src[idx]) * Ry_1
                        + (amb_temp - src[idx]) * Rz_1);
                } else if (r == row - 1 && c == 0) {
                    delta = Cap_1 * (g_power[idx]
                        + (src[idx + 1] - src[idx]) * Rx_1
                        + (src[idx - col] - src[idx]) * Ry_1
                        + (amb_temp - src[idx]) * Rz_1);
                } else if (r == 0) {
                    delta = Cap_1 * (g_power[idx]
                        + (src[idx + 1] + src[idx - 1] - 2.0f * src[idx]) * Rx_1
                        + (src[idx + col] - src[idx]) * Ry_1
                        + (amb_temp - src[idx]) * Rz_1);
                } else if (r == row - 1) {
                    delta = Cap_1 * (g_power[idx]
                        + (src[idx + 1] + src[idx - 1] - 2.0f * src[idx]) * Rx_1
                        + (src[idx - col] - src[idx]) * Ry_1
                        + (amb_temp - src[idx]) * Rz_1);
                } else if (c == 0) {
                    delta = Cap_1 * (g_power[idx]
                        + (src[idx + col] + src[idx - col] - 2.0f * src[idx]) * Ry_1
                        + (src[idx + 1] - src[idx]) * Rx_1
                        + (amb_temp - src[idx]) * Rz_1);
                } else if (c == col - 1) {
                    delta = Cap_1 * (g_power[idx]
                        + (src[idx + col] + src[idx - col] - 2.0f * src[idx]) * Ry_1
                        + (src[idx - 1] - src[idx]) * Rx_1
                        + (amb_temp - src[idx]) * Rz_1);
                } else {
                    delta = Cap_1 * (g_power[idx]
                        + (src[idx + col] + src[idx - col] - 2.0f * src[idx]) * Ry_1
                        + (src[idx + 1] + src[idx - 1] - 2.0f * src[idx]) * Rx_1
                        + (amb_temp - src[idx]) * Rz_1);
                }

                dst[idx] = src[idx] + delta;
            }
        }
        FLOAT* tmp = src;
        src = dst;
        dst = tmp;
    }

    memcpy(out_temp, src, n * sizeof(FLOAT));
}

void solution_free(void) {
    if (g_buf_a) free(g_buf_a);
    if (g_buf_b) free(g_buf_b);
    g_buf_a = NULL;
    g_buf_b = NULL;
    g_rows = g_cols = g_iters = 0;
    g_temp0 = NULL;
    g_power = NULL;
}

#ifdef __cplusplus
}
#endif
