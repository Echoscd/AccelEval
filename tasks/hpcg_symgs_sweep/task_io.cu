#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void solution_init(int n, const int *row_ptr, const int *col_idx,
                   const double *values, const int *diag_idx,
                   const double *rhs);
extern "C" void solution_compute(double *x_inout);
extern "C" void solution_free(void);

typedef struct {
    int n;
    const double *x_init;
    double *x_work;
} Ctx;

static int* get_tensor_int_local(const TaskData* data, const char* name) {
    for (int i = 0; i < data->num_inputs; ++i) {
        if (strcmp(data->inputs[i].name, name) == 0 && data->inputs[i].dtype == 0) {
            return (int*)data->inputs[i].data;
        }
    }
    return NULL;
}

static double* get_tensor_double_local(const TaskData* data, const char* name) {
    for (int i = 0; i < data->num_inputs; ++i) {
        if (strcmp(data->inputs[i].name, name) == 0 && data->inputs[i].dtype == 2) {
            return (double*)data->inputs[i].data;
        }
    }
    return NULL;
}

extern "C" void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    int n = (int)get_param(data, "n");
    const int *row_ptr = get_tensor_int_local(data, "row_ptr");
    const int *col_idx = get_tensor_int_local(data, "col_idx");
    const double *values = get_tensor_double_local(data, "values");
    const int *diag_idx = get_tensor_int_local(data, "diag_idx");
    const double *rhs = get_tensor_double_local(data, "rhs");
    const double *x_init = get_tensor_double_local(data, "x_init");
    if (!row_ptr || !col_idx || !values || !diag_idx || !rhs || !x_init) {
        fprintf(stderr, "[task_io] Missing HPCG SYMGS input tensor\n");
        return NULL;
    }
    solution_init(n, row_ptr, col_idx, values, diag_idx, rhs);
    Ctx *ctx = (Ctx*)calloc(1, sizeof(Ctx));
    if (!ctx) return NULL;
    ctx->n = n;
    ctx->x_init = x_init;
    ctx->x_work = (double*)malloc((size_t)n * sizeof(double));
    if (!ctx->x_work) {
        free(ctx);
        return NULL;
    }
    return ctx;
}

extern "C" void task_run(void* test_data) {
    Ctx *ctx = (Ctx*)test_data;
    memcpy(ctx->x_work, ctx->x_init, (size_t)ctx->n * sizeof(double));
    solution_compute(ctx->x_work);
}

extern "C" void task_write_output(void* test_data, const char* output_path) {
    Ctx *ctx = (Ctx*)test_data;
    FILE *f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->n; ++i) {
        fprintf(f, "%.17e\n", ctx->x_work[i]);
    }
    fclose(f);
}

extern "C" void task_cleanup(void* test_data) {
    Ctx *ctx = (Ctx*)test_data;
    if (!ctx) return;
    solution_free();
    free(ctx->x_work);
    free(ctx);
}
