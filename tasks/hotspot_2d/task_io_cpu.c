#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>

extern void solution_init(int rows, int cols, int iters,
                          const float* temp0,
                          const float* power);
extern void solution_compute(float* out_temp);
extern void solution_free(void);

typedef struct {
    int n;
    float* out_temp;
} TaskIOContext;

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    int rows = (int)get_param(data, "rows");
    int cols = (int)get_param(data, "cols");
    int iters = (int)get_param(data, "iters");
    const float* temp0 = get_tensor_float(data, "temp0");
    const float* power = get_tensor_float(data, "power");
    if (!temp0 || !power) {
        fprintf(stderr, "[task_io_cpu] Missing tensor data\n");
        return NULL;
    }
    solution_init(rows, cols, iters, temp0, power);
    TaskIOContext* ctx = (TaskIOContext*)calloc(1, sizeof(TaskIOContext));
    ctx->n = rows * cols;
    ctx->out_temp = (float*)calloc((size_t)ctx->n, sizeof(float));
    return ctx;
}

void task_run(void* test_data) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_compute(ctx->out_temp);
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
    solution_free();
    free(ctx->out_temp);
    free(ctx);
}
