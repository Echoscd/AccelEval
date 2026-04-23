// task_io_cpu.c — unified compute_only interface (auto-migrated)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void solution_compute(int N,
                             const float* xs,
                             const float* ys,
                             const float* zs,
                             float cell_size,
                             int grid_nx,
                             int grid_ny,
                             int grid_nz,
                             int num_cells,
                             int* sorted_indices,
                             int* cell_begin,
                             int* cell_end);

typedef struct {
    int N;
    const float* xs;
    const float* ys;
    const float* zs;
    float cell_size;
    int grid_nx;
    int grid_ny;
    int grid_nz;
    int num_cells;
    int* sorted_indices;
    int* cell_begin;
    int* cell_end;
} CellIdxContext;

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    CellIdxContext* ctx = (CellIdxContext*)calloc(1, sizeof(CellIdxContext));
    if (!ctx) return NULL;
    int cell_size_raw = (int)get_param(data, "cell_size_x1000000");
    ctx->N = (int)get_param(data, "N");
    ctx->xs = get_tensor_float(data, "xs");
    ctx->ys = get_tensor_float(data, "ys");
    ctx->zs = get_tensor_float(data, "zs");
    ctx->cell_size = (float)cell_size_raw / 1000000.0f;
    ctx->grid_nx = (int)get_param(data, "grid_nx");
    ctx->grid_ny = (int)get_param(data, "grid_ny");
    ctx->grid_nz = (int)get_param(data, "grid_nz");
    ctx->num_cells = ctx->grid_nx * ctx->grid_ny * ctx->grid_nz;

    if (!ctx->xs || !ctx->ys || !ctx->zs) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        free(ctx);
        return NULL;
    }
    ctx->sorted_indices = (int*)calloc((size_t)(ctx->N), sizeof(int));
    ctx->cell_begin = (int*)calloc((size_t)(ctx->num_cells), sizeof(int));
    ctx->cell_end = (int*)calloc((size_t)(ctx->num_cells), sizeof(int));
    return ctx;
}

void task_run(void* test_data) {
    CellIdxContext* ctx = (CellIdxContext*)test_data;
    solution_compute(ctx->N, ctx->xs, ctx->ys, ctx->zs, ctx->cell_size, ctx->grid_nx, ctx->grid_ny, ctx->grid_nz, ctx->num_cells, ctx->sorted_indices, ctx->cell_begin, ctx->cell_end);
}

void task_write_output(void* test_data, const char* output_path) {
    CellIdxContext* ctx = (CellIdxContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    int i;
    /* Write sorted indices */
    for (i = 0; i < ctx->N; i++)
        fprintf(f, "%d\n", ctx->sorted_indices[i]);
    /* Write cell_begin and cell_end */
    for (i = 0; i < ctx->num_cells; i++)
        fprintf(f, "%d %d\n", ctx->cell_begin[i], ctx->cell_end[i]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    CellIdxContext* ctx = (CellIdxContext*)test_data;
    free(ctx->sorted_indices);
    free(ctx->cell_begin);
    free(ctx->cell_end);
    free(ctx);
}
