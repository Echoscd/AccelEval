// task_io.cu — unified compute_only interface (auto-migrated)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

extern "C" void solution_compute(int num_points,
                             int num_spaces,
                             const float* points_xy,
                             const int* space_offsets,
                             float* results);

typedef struct {
    int num_points;
    int num_spaces;
    const float* points_xy;
    const int* space_offsets;
    float* results;
} HDContext;

extern "C" void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    HDContext* ctx = (HDContext*)calloc(1, sizeof(HDContext));
    if (!ctx) return NULL;
    ctx->num_points = (int)get_param(data, "num_points");
    ctx->num_spaces = (int)get_param(data, "num_spaces");
    ctx->points_xy = get_tensor_float(data, "points_xy");
    ctx->space_offsets = get_tensor_int(data, "space_offsets");

    if (!ctx->points_xy || !ctx->space_offsets) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        free(ctx);
        return NULL;
    }
    ctx->results = (float*)calloc((size_t)(ctx->num_spaces * ctx->num_spaces), sizeof(float));
    return ctx;
}

extern "C" void task_run(void* test_data) {
    HDContext* ctx = (HDContext*)test_data;
    solution_compute(ctx->num_points, ctx->num_spaces, ctx->points_xy, ctx->space_offsets, ctx->results);
}

extern "C" void task_write_output(void* test_data, const char* output_path) {
    HDContext* ctx = (HDContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    size_t total = (size_t)ctx->num_spaces * ctx->num_spaces;
    for (size_t i = 0; i < total; i++)
        fprintf(f, "%.6e\n", ctx->results[i]);
    fclose(f);
}

extern "C" void task_cleanup(void* test_data) {
    if (!test_data) return;
    HDContext* ctx = (HDContext*)test_data;
    free(ctx->results);
    free(ctx);
}
