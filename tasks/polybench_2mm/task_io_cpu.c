#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void solution_init(int ni, int nj, int nk, int nl,
                          int alpha_milli, int beta_milli,
                          const float *A, const float *B,
                          const float *C, const float *D0);
extern void solution_compute(float *D_out);
extern void solution_free(void);

typedef struct {
    int ni, nl;
    float *D_out;
} TwoMMTaskContext;

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
    int ni = (int)get_param(data, "ni");
    int nj = (int)get_param(data, "nj");
    int nk = (int)get_param(data, "nk");
    int nl = (int)get_param(data, "nl");
    int alpha_milli = (int)get_param(data, "alpha_milli");
    int beta_milli = (int)get_param(data, "beta_milli");
    const float *A = get_tensor_float_local(data, "A");
    const float *B = get_tensor_float_local(data, "B");
    const float *C = get_tensor_float_local(data, "C");
    const float *D0 = get_tensor_float_local(data, "D0");
    if (!A || !B || !C || !D0) {
        fprintf(stderr, "[task_io] Missing 2MM input tensor\n");
        return NULL;
    }
    solution_init(ni, nj, nk, nl, alpha_milli, beta_milli, A, B, C, D0);
    TwoMMTaskContext *ctx = (TwoMMTaskContext*)calloc(1, sizeof(TwoMMTaskContext));
    if (!ctx) return NULL;
    ctx->ni = ni;
    ctx->nl = nl;
    ctx->D_out = (float*)malloc((size_t)ni * (size_t)nl * sizeof(float));
    if (!ctx->D_out) {
        free(ctx);
        return NULL;
    }
    return ctx;
}

void task_run(void* test_data) {
    TwoMMTaskContext *ctx = (TwoMMTaskContext*)test_data;
    solution_compute(ctx->D_out);
}

void task_write_output(void* test_data, const char* output_path) {
    TwoMMTaskContext *ctx = (TwoMMTaskContext*)test_data;
    FILE *f = fopen(output_path, "w");
    if (!f) return;
    size_t total = (size_t)ctx->ni * (size_t)ctx->nl;
    for (size_t i = 0; i < total; ++i) {
        fprintf(f, "%.8e\n", ctx->D_out[i]);
    }
    fclose(f);
}

void task_cleanup(void* test_data) {
    TwoMMTaskContext *ctx = (TwoMMTaskContext*)test_data;
    if (!ctx) return;
    solution_free();
    free(ctx->D_out);
    free(ctx);
}
