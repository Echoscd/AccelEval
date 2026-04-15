// Adapted from the Rodinia Pathfinder benchmark core recurrence
// Source repo: https://github.com/yuhc/gpu-rodinia
// Original file: openmp/pathfinder/pathfinder.cpp

#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

static int g_rows = 0;
static int g_cols = 0;
static const int* g_wall = NULL;
static int* g_prev = NULL;
static int* g_next = NULL;

static int i_min(int a, int b) {
    return (a <= b) ? a : b;
}

void solution_init(int rows, int cols, const int* wall) {
    g_rows = rows;
    g_cols = cols;
    g_wall = wall;

    if (g_prev) free(g_prev);
    if (g_next) free(g_next);
    g_prev = (int*)malloc((size_t)cols * sizeof(int));
    g_next = (int*)malloc((size_t)cols * sizeof(int));
}

void solution_compute(int* out_costs) {
    if (!g_wall || !g_prev || !g_next || g_rows <= 0 || g_cols <= 0 || !out_costs) {
        return;
    }

    memcpy(g_prev, g_wall, (size_t)g_cols * sizeof(int));

    if (g_rows == 1) {
        memcpy(out_costs, g_prev, (size_t)g_cols * sizeof(int));
        return;
    }

    int* src = g_prev;
    int* dst = g_next;

    for (int r = 1; r < g_rows; ++r) {
        const int* row = g_wall + (size_t)r * (size_t)g_cols;
        for (int c = 0; c < g_cols; ++c) {
            int best = src[c];
            if (c > 0) {
                best = i_min(best, src[c - 1]);
            }
            if (c + 1 < g_cols) {
                best = i_min(best, src[c + 1]);
            }
            dst[c] = row[c] + best;
        }
        int* tmp = src;
        src = dst;
        dst = tmp;
    }

    memcpy(out_costs, src, (size_t)g_cols * sizeof(int));
}

void solution_free(void) {
    if (g_prev) {
        free(g_prev);
        g_prev = NULL;
    }
    if (g_next) {
        free(g_next);
        g_next = NULL;
    }
    g_rows = 0;
    g_cols = 0;
    g_wall = NULL;
}

#ifdef __cplusplus
}
#endif
