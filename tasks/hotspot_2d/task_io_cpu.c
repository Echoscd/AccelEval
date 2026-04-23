// task_io_cpu.c — unified compute_only interface (auto-migrated)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void solution_compute(int rows,
                             int cols,
                             int iters,
                             const float* temp0,
                             const float* power,
                             float* out_temp);

typedef struct {
    int rows;
    int cols;
    int iters;
    const float* temp0;
    const float* power;
    int n;
    float* out_temp;
} TaskIOContext;

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    TaskIOContext* ctx = (TaskIOContext*)calloc(1, sizeof(TaskIOContext));
    if (!ctx) return NULL;
    ctx->rows = (int)get_param(data, "rows");
    ctx->cols = (int)get_param(data, "cols");
    ctx->iters = (int)get_param(data, "iters");
    ctx->temp0 = get_tensor_float(data, "temp0");
    ctx->power = get_tensor_float(data, "power");

    if (!ctx->temp0 || !ctx->power) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        free(ctx);
        return NULL;
    }
    ctx->n = ctx->rows * ctx->cols;
    ctx->out_temp = (float*)calloc((size_t)(ctx->n), sizeof(float));
    return ctx;
}

void task_run(void* test_data) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_compute(ctx->rows, ctx->cols, ctx->iters, ctx->temp0, ctx->power, ctx->out_temp);
}

void task_write_output(void* test_data, const char* output_path) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->n; ++i) {
        fprintf(f, "%.6e\n", ctx->out_temp[i]);
    }
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    free(ctx->out_temp);
    free(ctx);
}
