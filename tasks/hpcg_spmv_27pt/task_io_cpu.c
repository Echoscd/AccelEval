#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void solution_init(int n, const int *row_ptr, const int *col_idx,
                          const double *values, const double *x);
extern void solution_compute(double *y_out);
extern void solution_free(void);

typedef struct {
    int n;
    double *y_out;
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
    int n = (int)get_param(data, "n");
    const int *row_ptr = get_tensor_int_local(data, "row_ptr");
    const int *col_idx = get_tensor_int_local(data, "col_idx");
    const double *values = get_tensor_double_local(data, "values");
    const double *x = get_tensor_double_local(data, "x");
    if (!row_ptr || !col_idx || !values || !x) {
        fprintf(stderr, "[task_io] Missing HPCG SpMV input tensor\n");
        return NULL;
    }
    solution_init(n, row_ptr, col_idx, values, x);
    Ctx *ctx = (Ctx*)calloc(1, sizeof(Ctx));
    if (!ctx) return NULL;
    ctx->n = n;
    ctx->y_out = (double*)malloc((size_t)n * sizeof(double));
    if (!ctx->y_out) {
        free(ctx);
        return NULL;
    }
    return ctx;
}

void task_run(void* test_data) {
    Ctx *ctx = (Ctx*)test_data;
    solution_compute(ctx->y_out);
}

void task_write_output(void* test_data, const char* output_path) {
    Ctx *ctx = (Ctx*)test_data;
    FILE *f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->n; ++i) {
        fprintf(f, "%.17e\n", ctx->y_out[i]);
    }
    fclose(f);
}

void task_cleanup(void* test_data) {
    Ctx *ctx = (Ctx*)test_data;
    if (!ctx) return;
    solution_free();
    free(ctx->y_out);
    free(ctx);
}
