#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void solution_init(int ni, int nj, int nk, int nl, int nm,
                               const float *A, const float *B,
                               const float *C, const float *D);
extern "C" void solution_compute(float *G_out);
extern "C" void solution_free(void);

typedef struct {
    int ni;
    int nl;
    float *G_out;
} ThreeMMTaskContext;

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
    int ni = (int)get_param(data, "ni");
    int nj = (int)get_param(data, "nj");
    int nk = (int)get_param(data, "nk");
    int nl = (int)get_param(data, "nl");
    int nm = (int)get_param(data, "nm");
    const float *A = get_tensor_float_local(data, "A");
    const float *B = get_tensor_float_local(data, "B");
    const float *C = get_tensor_float_local(data, "C");
    const float *D = get_tensor_float_local(data, "D");
    if (!A || !B || !C || !D) {
        fprintf(stderr, "[task_io] Missing PolyBench 3MM input tensor\n");
        return NULL;
    }
    solution_init(ni, nj, nk, nl, nm, A, B, C, D);
    ThreeMMTaskContext *ctx = (ThreeMMTaskContext*)calloc(1, sizeof(ThreeMMTaskContext));
    if (!ctx) return NULL;
    ctx->ni = ni;
    ctx->nl = nl;
    ctx->G_out = (float*)malloc((size_t)ni * (size_t)nl * sizeof(float));
    if (!ctx->G_out) {
        free(ctx);
        return NULL;
    }
    return ctx;
}

extern "C" void task_run(void* test_data) {
    ThreeMMTaskContext *ctx = (ThreeMMTaskContext*)test_data;
    solution_compute(ctx->G_out);
}

extern "C" void task_write_output(void* test_data, const char* output_path) {
    ThreeMMTaskContext *ctx = (ThreeMMTaskContext*)test_data;
    FILE *f = fopen(output_path, "w");
    if (!f) return;
    const size_t total = (size_t)ctx->ni * (size_t)ctx->nl;
    for (size_t idx = 0; idx < total; ++idx) {
        fprintf(f, "%.8e\n", ctx->G_out[idx]);
    }
    fclose(f);
}

extern "C" void task_cleanup(void* test_data) {
    ThreeMMTaskContext *ctx = (ThreeMMTaskContext*)test_data;
    if (!ctx) return;
    solution_free();
    free(ctx->G_out);
    free(ctx);
}
