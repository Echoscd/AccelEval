#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void solution_init(int n,
                               const float *A,
                               const float *u1,
                               const float *v1,
                               const float *u2,
                               const float *v2,
                               const float *y,
                               const float *z);
extern "C" void solution_compute(float *w_out);
extern "C" void solution_free(void);

typedef struct {
    int n;
    float *w_out;
} GemverTaskContext;

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

extern "C" void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    int n = (int)get_param(data, "n");
    const float *A  = get_tensor_float_local(data, "A");
    const float *u1 = get_tensor_float_local(data, "u1");
    const float *v1 = get_tensor_float_local(data, "v1");
    const float *u2 = get_tensor_float_local(data, "u2");
    const float *v2 = get_tensor_float_local(data, "v2");
    const float *y  = get_tensor_float_local(data, "y");
    const float *z  = get_tensor_float_local(data, "z");
    if (!A || !u1 || !v1 || !u2 || !v2 || !y || !z) {
        fprintf(stderr, "[task_io] Missing PolyBench GEMVER input tensor\n");
        return NULL;
    }
    solution_init(n, A, u1, v1, u2, v2, y, z);
    GemverTaskContext *ctx = (GemverTaskContext*)calloc(1, sizeof(GemverTaskContext));
    if (!ctx) return NULL;
    ctx->n = n;
    ctx->w_out = (float*)malloc((size_t)n * sizeof(float));
    if (!ctx->w_out) {
        free(ctx);
        return NULL;
    }
    return ctx;
}

extern "C" void task_run(void* test_data) {
    GemverTaskContext *ctx = (GemverTaskContext*)test_data;
    solution_compute(ctx->w_out);
}

extern "C" void task_write_output(void* test_data, const char* output_path) {
    GemverTaskContext *ctx = (GemverTaskContext*)test_data;
    FILE *f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->n; ++i) {
        fprintf(f, "%.8e\n", ctx->w_out[i]);
    }
    fclose(f);
}

extern "C" void task_cleanup(void* test_data) {
    GemverTaskContext *ctx = (GemverTaskContext*)test_data;
    if (!ctx) return;
    solution_free();
    free(ctx->w_out);
    free(ctx);
}
