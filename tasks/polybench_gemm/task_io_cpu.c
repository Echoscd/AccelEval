#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void solution_init(int ni, int nj, int nk,
                          int alpha_milli, int beta_milli,
                          const float *A, const float *B, const float *C0);
extern void solution_compute(float *C_out);
extern void solution_free(void);

typedef struct {
    int ni, nj;
    float *C_out;
} GemmTaskContext;

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
    int alpha_milli = (int)get_param(data, "alpha_milli");
    int beta_milli = (int)get_param(data, "beta_milli");
    const float *A = get_tensor_float_local(data, "A");
    const float *B = get_tensor_float_local(data, "B");
    const float *C0 = get_tensor_float_local(data, "C0");
    if (!A || !B || !C0) {
        fprintf(stderr, "[task_io] Missing GEMM input tensor\n");
        return NULL;
    }
    solution_init(ni, nj, nk, alpha_milli, beta_milli, A, B, C0);
    GemmTaskContext *ctx = (GemmTaskContext*)calloc(1, sizeof(GemmTaskContext));
    if (!ctx) return NULL;
    ctx->ni = ni;
    ctx->nj = nj;
    ctx->C_out = (float*)malloc((size_t)ni * (size_t)nj * sizeof(float));
    if (!ctx->C_out) {
        free(ctx);
        return NULL;
    }
    return ctx;
}

void task_run(void* test_data) {
    GemmTaskContext *ctx = (GemmTaskContext*)test_data;
    solution_compute(ctx->C_out);
}

void task_write_output(void* test_data, const char* output_path) {
    GemmTaskContext *ctx = (GemmTaskContext*)test_data;
    FILE *f = fopen(output_path, "w");
    if (!f) return;
    size_t total = (size_t)ctx->ni * (size_t)ctx->nj;
    for (size_t i = 0; i < total; ++i) {
        fprintf(f, "%.8e\n", ctx->C_out[i]);
    }
    fclose(f);
}

void task_cleanup(void* test_data) {
    GemmTaskContext *ctx = (GemmTaskContext*)test_data;
    if (!ctx) return;
    solution_free();
    free(ctx->C_out);
    free(ctx);
}
