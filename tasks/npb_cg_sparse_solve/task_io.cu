// task_io.cu — unified compute_only interface (auto-migrated)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

extern "C" void solution_compute(int n,
                             int nnz,
                             int max_iters,
                             int tol_exp,
                             const int * row_ptr,
                             const int * col_idx,
                             const double * values,
                             const double * b,
                             double * x_out);

typedef struct {
    int n;
    int nnz;
    int max_iters;
    int tol_exp;
    const int * row_ptr;
    const int * col_idx;
    const double * values;
    const double * b;
    double * x_out;
} CGTaskContext;

static double* get_tensor_double_local(const TaskData* data, const char* name) {
    if (!data || !name) return NULL;
    for (int i = 0; i < data->num_inputs; ++i) {
        if (strcmp(data->inputs[i].name, name) == 0) {
            if (data->inputs[i].dtype != 2) return NULL;
            return (double*)data->inputs[i].data;
        }
    }
    return NULL;
}

extern "C" void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    CGTaskContext* ctx = (CGTaskContext*)calloc(1, sizeof(CGTaskContext));
    if (!ctx) return NULL;
    ctx->n = (int)get_param(data, "n");
    ctx->nnz = (int)get_param(data, "nnz");
    ctx->max_iters = (int)get_param(data, "max_iters");
    ctx->tol_exp = (int)get_param(data, "tol_exp");
    ctx->row_ptr = get_tensor_int(data, "row_ptr");
    ctx->col_idx = get_tensor_int(data, "col_idx");
    ctx->values = get_tensor_double_local(data, "values");
    ctx->b = get_tensor_double_local(data, "b");

    if (!ctx->row_ptr || !ctx->col_idx || !ctx->values || !ctx->b) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        free(ctx);
        return NULL;
    }
    ctx->x_out = (double*)calloc((size_t)(ctx->n), sizeof(double));
    return ctx;
}

extern "C" void task_run(void* test_data) {
    CGTaskContext* ctx = (CGTaskContext*)test_data;
    solution_compute(ctx->n, ctx->nnz, ctx->max_iters, ctx->tol_exp, ctx->row_ptr, ctx->col_idx, ctx->values, ctx->b, ctx->x_out);
}

extern "C" void task_write_output(void* test_data, const char* output_path) {
    CGTaskContext* ctx = (CGTaskContext*)test_data;
    FILE *f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->n; ++i) {
        fprintf(f, "%.15e\n", ctx->x_out[i]);
    }
    fclose(f);
}

extern "C" void task_cleanup(void* test_data) {
    if (!test_data) return;
    CGTaskContext* ctx = (CGTaskContext*)test_data;
    free(ctx->x_out);
    free(ctx);
}
