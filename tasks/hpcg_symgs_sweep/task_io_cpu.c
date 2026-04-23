// task_io_cpu.c — unified compute_only interface (auto-migrated)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void solution_compute(int n,
                             const int * row_ptr,
                             const int * col_idx,
                             const double * values,
                             const int * diag_idx,
                             const double * rhs,
                             double * x_inout);

typedef struct {
    int n;
    const int * row_ptr;
    const int * col_idx;
    const double * values;
    const int * diag_idx;
    const double * rhs;
    const double * x_init;
    double * x_inout;
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

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    Ctx* ctx = (Ctx*)calloc(1, sizeof(Ctx));
    if (!ctx) return NULL;
    ctx->n = (int)get_param(data, "n");
    ctx->row_ptr = get_tensor_int_local(data, "row_ptr");
    ctx->col_idx = get_tensor_int_local(data, "col_idx");
    ctx->values = get_tensor_double_local(data, "values");
    ctx->diag_idx = get_tensor_int_local(data, "diag_idx");
    ctx->rhs = get_tensor_double_local(data, "rhs");
    ctx->x_init = get_tensor_double_local(data, "x_init");

    if (!ctx->row_ptr || !ctx->col_idx || !ctx->values || !ctx->diag_idx || !ctx->rhs || !ctx->x_init) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        free(ctx);
        return NULL;
    }
    ctx->x_inout = (double*)calloc((size_t)(ctx->n), sizeof(double));
    return ctx;
}

void task_run(void* test_data) {
    Ctx* ctx = (Ctx*)test_data;
    memcpy((void*)ctx->x_inout, ctx->x_init, (size_t)ctx->n * sizeof(double));
    solution_compute(ctx->n, ctx->row_ptr, ctx->col_idx, ctx->values, ctx->diag_idx, ctx->rhs, ctx->x_inout);
}

void task_write_output(void* test_data, const char* output_path) {
    Ctx* ctx = (Ctx*)test_data;
    FILE *f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->n; ++i) {
        fprintf(f, "%.17e\n", ctx->x_inout[i]);
    }
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    Ctx* ctx = (Ctx*)test_data;
    free(ctx->x_inout);
    free(ctx);
}
