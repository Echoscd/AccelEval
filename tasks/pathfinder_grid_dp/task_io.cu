// task_io.cu — unified compute_only interface (auto-migrated)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

extern "C" void solution_compute(int rows,
                             int cols,
                             const int* wall,
                             int* out_costs);

typedef struct {
    int rows;
    int cols;
    const int* wall;
    int* out_costs;
} TaskIOContext;

extern "C" void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    TaskIOContext* ctx = (TaskIOContext*)calloc(1, sizeof(TaskIOContext));
    if (!ctx) return NULL;
    ctx->rows = (int)get_param(data, "rows");
    ctx->cols = (int)get_param(data, "cols");
    ctx->wall = get_tensor_int(data, "wall");

    if (!ctx->wall) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        free(ctx);
        return NULL;
    }
    ctx->out_costs = (int*)calloc((size_t)(ctx->cols), sizeof(int));
    return ctx;
}

extern "C" void task_run(void* test_data) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_compute(ctx->rows, ctx->cols, ctx->wall, ctx->out_costs);
}

extern "C" void task_write_output(void* test_data, const char* output_path) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int c = 0; c < ctx->cols; ++c) {
        fprintf(f, "%d\n", ctx->out_costs[c]);
    }
    fclose(f);
}

extern "C" void task_cleanup(void* test_data) {
    if (!test_data) return;
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    free(ctx->out_costs);
    free(ctx);
}
