// task_io_cpu.c -- spmv_csr CPU I/O adapter
// Computes answer = A^T * vector using CSC matrix.

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>

extern void solution_init(int num_rows, int num_cols,
                          const int* col_ptrs, const int* row_indices,
                          const float* values, const float* vector);
extern void solution_compute(int num_cols, float* answer);
extern void solution_free(void);

// Weak default: LLM does not need to implement solution_free
__attribute__((weak)) void solution_free(void) { }

typedef struct {
    int    num_cols;
    float* answer;
} SpMVContext;

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    int num_rows = (int)get_param(data, "num_rows");
    int num_cols = (int)get_param(data, "num_cols");

    const int*   col_ptrs    = get_tensor_int(data, "col_ptrs");
    const int*   row_indices = get_tensor_int(data, "row_indices");
    const float* values      = get_tensor_float(data, "values");
    const float* vector      = get_tensor_float(data, "vector");
    if (!col_ptrs || !row_indices || !values || !vector) {
        fprintf(stderr, "[task_io] missing required tensors for spmv_csr\n");
        return NULL;
    }

    SpMVContext* ctx = (SpMVContext*)calloc(1, sizeof(SpMVContext));
    ctx->num_cols = num_cols;
    ctx->answer   = (float*)calloc((size_t)num_cols, sizeof(float));

    solution_init(num_rows, num_cols, col_ptrs, row_indices, values, vector);
    return ctx;
}

void task_run(void* test_data) {
    SpMVContext* ctx = (SpMVContext*)test_data;
    solution_compute(ctx->num_cols, ctx->answer);
}

void task_write_output(void* test_data, const char* output_path) {
    SpMVContext* ctx = (SpMVContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int j = 0; j < ctx->num_cols; j++)
        fprintf(f, "%.8e\n", ctx->answer[j]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    SpMVContext* ctx = (SpMVContext*)test_data;
    solution_free();
    free(ctx->answer);
    free(ctx);
}
