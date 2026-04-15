#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void solution_init(int n, int nnz, int max_iters, int tol_exp,
                          const int *row_ptr, const int *col_idx,
                          const double *values, const double *b);
extern "C" void solution_compute(double *x_out);
extern "C" void solution_free(void);

typedef struct {
    int n;
    double *x_out;
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
    int n = (int)get_param(data, "n");
    int nnz = (int)get_param(data, "nnz");
    int max_iters = (int)get_param(data, "max_iters");
    int tol_exp = (int)get_param(data, "tol_exp");
    const int *row_ptr = get_tensor_int(data, "row_ptr");
    const int *col_idx = get_tensor_int(data, "col_idx");
    const double *values = get_tensor_double_local(data, "values");
    const double *b = get_tensor_double_local(data, "b");
    if (!row_ptr || !col_idx || !values || !b) {
        fprintf(stderr, "[task_io] Missing CG input tensor\n");
        return NULL;
    }
    solution_init(n, nnz, max_iters, tol_exp, row_ptr, col_idx, values, b);
    CGTaskContext *ctx = (CGTaskContext*)calloc(1, sizeof(CGTaskContext));
    ctx->n = n;
    ctx->x_out = (double*)malloc((size_t)n * sizeof(double));
    return ctx;
}

extern "C" void task_run(void* test_data) {
    CGTaskContext *ctx = (CGTaskContext*)test_data;
    solution_compute(ctx->x_out);
}

extern "C" void task_write_output(void* test_data, const char* output_path) {
    CGTaskContext *ctx = (CGTaskContext*)test_data;
    FILE *f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->n; ++i) {
        fprintf(f, "%.15e\n", ctx->x_out[i]);
    }
    fclose(f);
}

extern "C" void task_cleanup(void* test_data) {
    if (!test_data) return;
    CGTaskContext *ctx = (CGTaskContext*)test_data;
    solution_free();
    free(ctx->x_out);
    free(ctx);
}
