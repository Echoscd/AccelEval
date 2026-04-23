// task_io_cpu.c — unified compute_only interface (auto-migrated)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void solution_compute(int N,
                             const int* types,
                             const float* strikes,
                             const float* spots,
                             const float* qs,
                             const float* rs,
                             const float* ts,
                             const float* vols,
                             float* prices);

typedef struct {
    int N;
    const int* types;
    const float* strikes;
    const float* spots;
    const float* qs;
    const float* rs;
    const float* ts;
    const float* vols;
    float* prices;
} BSContext;

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    BSContext* ctx = (BSContext*)calloc(1, sizeof(BSContext));
    if (!ctx) return NULL;
    ctx->N = (int)get_param(data, "N");
    ctx->types = get_tensor_int(data, "types");
    ctx->strikes = get_tensor_float(data, "strikes");
    ctx->spots = get_tensor_float(data, "spots");
    ctx->qs = get_tensor_float(data, "qs");
    ctx->rs = get_tensor_float(data, "rs");
    ctx->ts = get_tensor_float(data, "ts");
    ctx->vols = get_tensor_float(data, "vols");

    if (!ctx->types || !ctx->strikes || !ctx->spots || !ctx->qs || !ctx->rs || !ctx->ts || !ctx->vols) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        free(ctx);
        return NULL;
    }
    ctx->prices = (float*)calloc((size_t)(ctx->N), sizeof(float));
    return ctx;
}

void task_run(void* test_data) {
    BSContext* ctx = (BSContext*)test_data;
    solution_compute(ctx->N, ctx->types, ctx->strikes, ctx->spots, ctx->qs, ctx->rs, ctx->ts, ctx->vols, ctx->prices);
}

void task_write_output(void* test_data, const char* output_path) {
    BSContext* ctx = (BSContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->N; i++)
        fprintf(f, "%.6f\n", ctx->prices[i]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    BSContext* ctx = (BSContext*)test_data;
    free(ctx->prices);
    free(ctx);
}
