// task_io.cu — unified compute_only interface (auto-migrated)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

extern "C" void solution_compute(int num_rows,
                             int num_cols,
                             const int* col_ptrs,
                             const int* row_indices,
                             const float* values,
                             const float* vector,
                             float* answer);

typedef struct {
    int num_rows;
    int num_cols;
    const int* col_ptrs;
    const int* row_indices;
    const float* values;
    const float* vector;
    float* answer;
} SpMVContext;

extern "C" void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    SpMVContext* ctx = (SpMVContext*)calloc(1, sizeof(SpMVContext));
    if (!ctx) return NULL;
    ctx->num_rows = (int)get_param(data, "num_rows");
    ctx->num_cols = (int)get_param(data, "num_cols");
    ctx->col_ptrs = get_tensor_int(data, "col_ptrs");
    ctx->row_indices = get_tensor_int(data, "row_indices");
    ctx->values = get_tensor_float(data, "values");
    ctx->vector = get_tensor_float(data, "vector");

    if (!ctx->col_ptrs || !ctx->row_indices || !ctx->values || !ctx->vector) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        free(ctx);
        return NULL;
    }
    ctx->answer = (float*)calloc((size_t)(ctx->num_cols), sizeof(float));
    return ctx;
}

extern "C" void task_run(void* test_data) {
    SpMVContext* ctx = (SpMVContext*)test_data;
    solution_compute(ctx->num_rows, ctx->num_cols, ctx->col_ptrs, ctx->row_indices, ctx->values, ctx->vector, ctx->answer);
}

extern "C" void task_write_output(void* test_data, const char* output_path) {
    SpMVContext* ctx = (SpMVContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int j = 0; j < ctx->num_cols; j++)
        fprintf(f, "%.8e\n", ctx->answer[j]);
    fclose(f);
}

extern "C" void task_cleanup(void* test_data) {
    if (!test_data) return;
    SpMVContext* ctx = (SpMVContext*)test_data;
    free(ctx->answer);
    free(ctx);
}
