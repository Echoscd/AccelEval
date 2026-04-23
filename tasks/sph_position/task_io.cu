// task_io.cu — unified compute_only interface (auto-migrated)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

extern "C" void solution_compute(int N,
                             const float* posxy_x,
                             const float* posxy_y,
                             const float* posz,
                             const float* movxy_x,
                             const float* movxy_y,
                             const float* movz,
                             float cell_size,
                             double* out_x,
                             double* out_y,
                             double* out_z,
                             int* out_cell);

typedef struct {
    int N;
    const float* posxy_x;
    const float* posxy_y;
    const float* posz;
    const float* movxy_x;
    const float* movxy_y;
    const float* movz;
    float cell_size;
    double* out_x;
    double* out_y;
    double* out_z;
    int* out_cell;
} SPHPosContext;

extern "C" void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    SPHPosContext* ctx = (SPHPosContext*)calloc(1, sizeof(SPHPosContext));
    if (!ctx) return NULL;
    const float* pos_x = get_tensor_float(data, "pos_x");
    const float* pos_y = get_tensor_float(data, "pos_y");
    const float* pos_z = get_tensor_float(data, "pos_z");
    const float* mov_x = get_tensor_float(data, "mov_x");
    const float* mov_y = get_tensor_float(data, "mov_y");
    const float* mov_z = get_tensor_float(data, "mov_z");
    ctx->N = (int)get_param(data, "N");
    ctx->posxy_x = get_tensor_float(data, "pos_x");
    ctx->posxy_y = get_tensor_float(data, "pos_y");
    ctx->posz = get_tensor_float(data, "pos_z");
    ctx->movxy_x = get_tensor_float(data, "mov_x");
    ctx->movxy_y = get_tensor_float(data, "mov_y");
    ctx->movz = get_tensor_float(data, "mov_z");
    ctx->cell_size = (double)get_param(data, "cell_size_x1000000") / 1000000.0f;

    if (!ctx->posxy_x || !ctx->posxy_y || !ctx->posz || !ctx->movxy_x || !ctx->movxy_y || !ctx->movz) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        free(ctx);
        return NULL;
    }
    ctx->out_x = (double*)calloc((size_t)(ctx->N), sizeof(double));
    ctx->out_y = (double*)calloc((size_t)(ctx->N), sizeof(double));
    ctx->out_z = (double*)calloc((size_t)(ctx->N), sizeof(double));
    ctx->out_cell = (int*)calloc((size_t)(ctx->N), sizeof(int));
    return ctx;
}

extern "C" void task_run(void* test_data) {
    SPHPosContext* ctx = (SPHPosContext*)test_data;
    solution_compute(ctx->N, ctx->posxy_x, ctx->posxy_y, ctx->posz, ctx->movxy_x, ctx->movxy_y, ctx->movz, ctx->cell_size, ctx->out_x, ctx->out_y, ctx->out_z, ctx->out_cell);
}

extern "C" void task_write_output(void* test_data, const char* output_path) {
    SPHPosContext* ctx = (SPHPosContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->N; i++)
        fprintf(f, "%.10f %.10f %.10f %d\n",
                ctx->out_x[i], ctx->out_y[i], ctx->out_z[i], ctx->out_cell[i]);
    fclose(f);
}

extern "C" void task_cleanup(void* test_data) {
    if (!test_data) return;
    SPHPosContext* ctx = (SPHPosContext*)test_data;
    free(ctx->out_x);
    free(ctx->out_y);
    free(ctx->out_z);
    free(ctx->out_cell);
    free(ctx);
}
