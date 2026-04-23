// task_io_cpu.c — unified compute_only interface (auto-migrated)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void solution_compute(int n,
                             int max_iters,
                             const int * in_row_ptr,
                             const int * in_col_idx,
                             const int * out_degree,
                             float * scores_out);

typedef struct {
    int n;
    int max_iters;
    const int * in_row_ptr;
    const int * in_col_idx;
    const int * out_degree;
    float * scores_out;
} PRTaskContext;

static int* get_tensor_int_local(const TaskData* data, const char* name) {
    if (!data || !name) return NULL;
    for (int i = 0; i < data->num_inputs; ++i) {
        if (strcmp(data->inputs[i].name, name) == 0) {
            if (data->inputs[i].dtype != 0) return NULL;
            return (int*)data->inputs[i].data;
        }
    }
    return NULL;
}

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    PRTaskContext* ctx = (PRTaskContext*)calloc(1, sizeof(PRTaskContext));
    if (!ctx) return NULL;
    ctx->n = (int)get_param(data, "n");
    ctx->max_iters = (int)get_param(data, "max_iters");
    ctx->in_row_ptr = get_tensor_int_local(data, "in_row_ptr");
    ctx->in_col_idx = get_tensor_int_local(data, "in_col_idx");
    ctx->out_degree = get_tensor_int_local(data, "out_degree");

    if (!ctx->in_row_ptr || !ctx->in_col_idx || !ctx->out_degree) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        free(ctx);
        return NULL;
    }
    ctx->scores_out = (float*)calloc((size_t)(ctx->n), sizeof(float));
    return ctx;
}

void task_run(void* test_data) {
    PRTaskContext* ctx = (PRTaskContext*)test_data;
    solution_compute(ctx->n, ctx->max_iters, ctx->in_row_ptr, ctx->in_col_idx, ctx->out_degree, ctx->scores_out);
}

void task_write_output(void* test_data, const char* output_path) {
    PRTaskContext* ctx = (PRTaskContext*)test_data;
    FILE *f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->n; ++i) {
        fprintf(f, "%.8e\n", ctx->scores_out[i]);
    }
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    PRTaskContext* ctx = (PRTaskContext*)test_data;
    free(ctx->scores_out);
    free(ctx);
}
