// task_io_cpu.c — unified compute_only interface (auto-migrated)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void solution_compute(int n,
                             int iters,
                             int omega_milli,
                             const double * u0,
                             const double * rhs,
                             double * residual_out);

typedef struct {
    int n;
    int iters;
    int omega_milli;
    const double * u0;
    const double * rhs;
    double * residual_out;
} LUTaskContext;

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

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    LUTaskContext* ctx = (LUTaskContext*)calloc(1, sizeof(LUTaskContext));
    if (!ctx) return NULL;
    ctx->n = (int)get_param(data, "n");
    ctx->iters = (int)get_param(data, "iters");
    ctx->omega_milli = (int)get_param(data, "omega_milli");
    ctx->u0 = get_tensor_double_local(data, "u0");
    ctx->rhs = get_tensor_double_local(data, "rhs");

    if (!ctx->u0 || !ctx->rhs) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        free(ctx);
        return NULL;
    }
    ctx->residual_out = (double*)calloc((size_t)(5), sizeof(double));
    return ctx;
}

void task_run(void* test_data) {
    LUTaskContext* ctx = (LUTaskContext*)test_data;
    solution_compute(ctx->n, ctx->iters, ctx->omega_milli, ctx->u0, ctx->rhs, ctx->residual_out);
}

void task_write_output(void* test_data, const char* output_path) {
    LUTaskContext* ctx = (LUTaskContext*)test_data;
    FILE *f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < 5; ++i) fprintf(f, "%.15e\n", ctx->residual_out[i]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    LUTaskContext* ctx = (LUTaskContext*)test_data;
    free(ctx->residual_out);
    free(ctx);
}
