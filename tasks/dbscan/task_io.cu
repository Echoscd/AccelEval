// task_io.cu — unified compute_only interface (auto-migrated)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

extern "C" void solution_compute(int N,
                             const float* xs,
                             const float* ys,
                             float eps,
                             int minPts,
                             int* labels);

typedef struct {
    int N;
    const float* xs;
    const float* ys;
    float eps;
    int minPts;
    int* labels;
} DBSCANContext;

extern "C" void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    DBSCANContext* ctx = (DBSCANContext*)calloc(1, sizeof(DBSCANContext));
    if (!ctx) return NULL;
    int eps_x10000 = (int)get_param(data, "eps_x10000");
    ctx->N = (int)get_param(data, "N");
    ctx->xs = get_tensor_float(data, "xs");
    ctx->ys = get_tensor_float(data, "ys");
    ctx->eps = (float)eps_x10000 / 10000.0f;
    ctx->minPts = (int)get_param(data, "minPts");

    if (!ctx->xs || !ctx->ys) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        free(ctx);
        return NULL;
    }
    ctx->labels = (int*)calloc((size_t)(ctx->N), sizeof(int));
    return ctx;
}

extern "C" void task_run(void* test_data) {
    DBSCANContext* ctx = (DBSCANContext*)test_data;
    solution_compute(ctx->N, ctx->xs, ctx->ys, ctx->eps, ctx->minPts, ctx->labels);
}

extern "C" void task_write_output(void* test_data, const char* output_path) {
    DBSCANContext* ctx = (DBSCANContext*)test_data;
    FILE* fp = fopen(output_path, "w");
    if (!fp) return;
    for (int i = 0; i < ctx->N; i++) {
        fprintf(fp, "%d\n", ctx->labels[i]);
    }
    fclose(fp);
}

extern "C" void task_cleanup(void* test_data) {
    if (!test_data) return;
    DBSCANContext* ctx = (DBSCANContext*)test_data;
    free(ctx->labels);
    free(ctx);
}
