#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void solution_init(int n, int tsteps, const float *A0, const float *B0);
extern void solution_compute(float *A_out);
extern void solution_free(void);

typedef struct {
    int n;
    float *A_out;
} Jacobi2DTaskContext;

static float* get_tensor_float_local(const TaskData* data, const char* name) {
    if (!data || !name) return NULL;
    for (int i = 0; i < data->num_inputs; ++i) {
        if (strcmp(data->inputs[i].name, name) == 0) {
            if (data->inputs[i].dtype != 1) return NULL;
            return (float*)data->inputs[i].data;
        }
    }
    return NULL;
}

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    int n = (int)get_param(data, "n");
    int tsteps = (int)get_param(data, "tsteps");
    const float *A0 = get_tensor_float_local(data, "A0");
    const float *B0 = get_tensor_float_local(data, "B0");
    if (!A0 || !B0) {
        fprintf(stderr, "[task_io] Missing Jacobi-2D input tensor\n");
        return NULL;
    }
    solution_init(n, tsteps, A0, B0);
    Jacobi2DTaskContext *ctx = (Jacobi2DTaskContext*)calloc(1, sizeof(Jacobi2DTaskContext));
    if (!ctx) return NULL;
    ctx->n = n;
    ctx->A_out = (float*)malloc((size_t)n * (size_t)n * sizeof(float));
    if (!ctx->A_out) {
        free(ctx);
        return NULL;
    }
    return ctx;
}

void task_run(void* test_data) {
    Jacobi2DTaskContext *ctx = (Jacobi2DTaskContext*)test_data;
    solution_compute(ctx->A_out);
}

void task_write_output(void* test_data, const char* output_path) {
    Jacobi2DTaskContext *ctx = (Jacobi2DTaskContext*)test_data;
    FILE *f = fopen(output_path, "w");
    if (!f) return;
    const size_t total = (size_t)ctx->n * (size_t)ctx->n;
    for (size_t idx = 0; idx < total; ++idx) {
        fprintf(f, "%.8e\n", ctx->A_out[idx]);
    }
    fclose(f);
}

void task_cleanup(void* test_data) {
    Jacobi2DTaskContext *ctx = (Jacobi2DTaskContext*)test_data;
    if (!ctx) return;
    solution_free();
    free(ctx->A_out);
    free(ctx);
}
