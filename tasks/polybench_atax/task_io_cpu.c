#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void solution_init(int m, int n, const float *A, const float *x);
extern void solution_compute(float *y_out);
extern void solution_free(void);

typedef struct {
    int n;
    float *y_out;
} AtaxTaskContext;

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
    int m = (int)get_param(data, "m");
    int n = (int)get_param(data, "n");
    const float *A = get_tensor_float_local(data, "A");
    const float *x = get_tensor_float_local(data, "x");
    if (!A || !x) {
        fprintf(stderr, "[task_io] Missing PolyBench ATAX input tensor\n");
        return NULL;
    }
    solution_init(m, n, A, x);
    AtaxTaskContext *ctx = (AtaxTaskContext*)calloc(1, sizeof(AtaxTaskContext));
    if (!ctx) return NULL;
    ctx->n = n;
    ctx->y_out = (float*)malloc((size_t)n * sizeof(float));
    if (!ctx->y_out) {
        free(ctx);
        return NULL;
    }
    return ctx;
}

void task_run(void* test_data) {
    AtaxTaskContext *ctx = (AtaxTaskContext*)test_data;
    solution_compute(ctx->y_out);
}

void task_write_output(void* test_data, const char* output_path) {
    AtaxTaskContext *ctx = (AtaxTaskContext*)test_data;
    FILE *f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->n; ++i) {
        fprintf(f, "%.8e\n", ctx->y_out[i]);
    }
    fclose(f);
}

void task_cleanup(void* test_data) {
    AtaxTaskContext *ctx = (AtaxTaskContext*)test_data;
    if (!ctx) return;
    solution_free();
    free(ctx->y_out);
    free(ctx);
}
