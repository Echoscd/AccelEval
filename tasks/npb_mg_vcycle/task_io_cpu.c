#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>

extern void solution_init(int interior_n, int num_cycles, int pre_smooth, int post_smooth,
                          int coarse_iters, const double* h_rhs);
extern void solution_compute(double* h_residual_norm_out);
extern void solution_free(void);

typedef struct {
    double residual_norm_out[1];
} MGContext;

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
    MGContext* ctx = (MGContext*)calloc(1, sizeof(MGContext));
    int interior_n = (int)get_param(data, "interior_n");
    int num_cycles = (int)get_param(data, "num_cycles");
    int pre_smooth = (int)get_param(data, "pre_smooth");
    int post_smooth = (int)get_param(data, "post_smooth");
    int coarse_iters = (int)get_param(data, "coarse_iters");
    solution_init(interior_n, num_cycles, pre_smooth, post_smooth, coarse_iters,
                  get_tensor_double_local(data, "rhs"));
    return ctx;
}

void task_run(void* test_data) {
    MGContext* ctx = (MGContext*)test_data;
    solution_compute(ctx->residual_norm_out);
}

void task_write_output(void* test_data, const char* output_path) {
    MGContext* ctx = (MGContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    fprintf(f, "%.15e\n", ctx->residual_norm_out[0]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    solution_free();
    free(test_data);
}
