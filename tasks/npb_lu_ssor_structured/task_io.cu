#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void solution_init(int n, int iters, int omega_milli,
                               const double *u0, const double *rhs);
extern "C" void solution_compute(double *residual_out);
extern "C" void solution_free(void);

typedef struct {
    double residuals[5];
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

extern "C" void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    int n = (int)get_param(data, "n");
    int iters = (int)get_param(data, "iters");
    int omega_milli = (int)get_param(data, "omega_milli");
    const double *u0 = get_tensor_double_local(data, "u0");
    const double *rhs = get_tensor_double_local(data, "rhs");
    if (!u0 || !rhs) {
        fprintf(stderr, "[task_io] Missing LU-SSOR input tensor\n");
        return NULL;
    }
    solution_init(n, iters, omega_milli, u0, rhs);
    LUTaskContext *ctx = (LUTaskContext*)calloc(1, sizeof(LUTaskContext));
    return ctx;
}

extern "C" void task_run(void* test_data) {
    LUTaskContext *ctx = (LUTaskContext*)test_data;
    solution_compute(ctx->residuals);
}

extern "C" void task_write_output(void* test_data, const char* output_path) {
    LUTaskContext *ctx = (LUTaskContext*)test_data;
    FILE *f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < 5; ++i) fprintf(f, "%.15e\n", ctx->residuals[i]);
    fclose(f);
}

extern "C" void task_cleanup(void* test_data) {
    if (!test_data) return;
    solution_free();
    free(test_data);
}
