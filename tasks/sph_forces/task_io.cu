// task_io.cu — unified compute_only interface (auto-migrated)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

extern "C" void solution_compute(int N,
                             const float* xs,
                             const float* ys,
                             const float* zs,
                             const float* vxs,
                             const float* vys,
                             const float* vzs,
                             const float* rhos,
                             const float* masses,
                             float h,
                             float cs0,
                             float rhop0,
                             float alpha_visc,
                             const int* cell_begin,
                             const int* cell_end,
                             const int* sorted_idx,
                             int grid_nx,
                             int grid_ny,
                             int grid_nz,
                             float cell_size,
                             float* ax,
                             float* ay,
                             float* az,
                             float* drhodt);

typedef struct {
    int N;
    const float* xs;
    const float* ys;
    const float* zs;
    const float* vxs;
    const float* vys;
    const float* vzs;
    const float* rhos;
    const float* masses;
    float h;
    float cs0;
    float rhop0;
    float alpha_visc;
    const int* cell_begin;
    const int* cell_end;
    const int* sorted_idx;
    int grid_nx;
    int grid_ny;
    int grid_nz;
    float cell_size;
    float* ax;
    float* ay;
    float* az;
    float* drhodt;
} SPHForceContext;

extern "C" void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    SPHForceContext* ctx = (SPHForceContext*)calloc(1, sizeof(SPHForceContext));
    if (!ctx) return NULL;
    ctx->N = (int)get_param(data, "N");
    ctx->xs = get_tensor_float(data, "xs");
    ctx->ys = get_tensor_float(data, "ys");
    ctx->zs = get_tensor_float(data, "zs");
    ctx->vxs = get_tensor_float(data, "vxs");
    ctx->vys = get_tensor_float(data, "vys");
    ctx->vzs = get_tensor_float(data, "vzs");
    ctx->rhos = get_tensor_float(data, "rhos");
    ctx->masses = get_tensor_float(data, "masses");
    ctx->h = (float)get_param(data, "h_x1000000") / 1000000.0f;
    ctx->cs0 = (float)get_param(data, "cs0_x10000") / 10000.0f;
    ctx->rhop0 = (float)get_param(data, "rhop0_x100") / 100.0f;
    ctx->alpha_visc = (float)get_param(data, "alpha_visc_x10000") / 10000.0f;
    ctx->cell_begin = get_tensor_int(data, "cell_begin");
    ctx->cell_end = get_tensor_int(data, "cell_end");
    ctx->sorted_idx = get_tensor_int(data, "sorted_idx");
    ctx->grid_nx = (int)get_param(data, "grid_nx");
    ctx->grid_ny = (int)get_param(data, "grid_ny");
    ctx->grid_nz = (int)get_param(data, "grid_nz");
    ctx->cell_size = (float)get_param(data, "cell_size_x1000000") / 1000000.0f;

    if (!ctx->xs || !ctx->ys || !ctx->zs || !ctx->vxs || !ctx->vys || !ctx->vzs || !ctx->rhos || !ctx->masses || !ctx->cell_begin || !ctx->cell_end || !ctx->sorted_idx) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        free(ctx);
        return NULL;
    }
    ctx->ax = (float*)calloc((size_t)(ctx->N), sizeof(float));
    ctx->ay = (float*)calloc((size_t)(ctx->N), sizeof(float));
    ctx->az = (float*)calloc((size_t)(ctx->N), sizeof(float));
    ctx->drhodt = (float*)calloc((size_t)(ctx->N), sizeof(float));
    return ctx;
}

extern "C" void task_run(void* test_data) {
    SPHForceContext* ctx = (SPHForceContext*)test_data;
    solution_compute(ctx->N, ctx->xs, ctx->ys, ctx->zs, ctx->vxs, ctx->vys, ctx->vzs, ctx->rhos, ctx->masses, ctx->h, ctx->cs0, ctx->rhop0, ctx->alpha_visc, ctx->cell_begin, ctx->cell_end, ctx->sorted_idx, ctx->grid_nx, ctx->grid_ny, ctx->grid_nz, ctx->cell_size, ctx->ax, ctx->ay, ctx->az, ctx->drhodt);
}

extern "C" void task_write_output(void* test_data, const char* output_path) {
    SPHForceContext* ctx = (SPHForceContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->N; i++)
        fprintf(f, "%.6e %.6e %.6e %.6e\n",
                ctx->ax[i], ctx->ay[i], ctx->az[i], ctx->drhodt[i]);
    fclose(f);
}

extern "C" void task_cleanup(void* test_data) {
    if (!test_data) return;
    SPHForceContext* ctx = (SPHForceContext*)test_data;
    free(ctx->ax);
    free(ctx->ay);
    free(ctx->az);
    free(ctx->drhodt);
    free(ctx);
}
