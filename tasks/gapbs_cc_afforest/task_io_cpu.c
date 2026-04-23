// task_io_cpu.c — unified compute_only interface (auto-migrated)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void solution_compute(int n,
                             int neighbor_rounds,
                             int num_samples,
                             const int * row_ptr,
                             const int * col_idx,
                             int * comp_out);

typedef struct {
    int n;
    int neighbor_rounds;
    int num_samples;
    const int * row_ptr;
    const int * col_idx;
    int * comp_out;
} CCTaskContext;

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
    CCTaskContext* ctx = (CCTaskContext*)calloc(1, sizeof(CCTaskContext));
    if (!ctx) return NULL;
    ctx->n = (int)get_param(data, "n");
    ctx->neighbor_rounds = (int)get_param(data, "neighbor_rounds");
    ctx->num_samples = (int)get_param(data, "num_samples");
    ctx->row_ptr = get_tensor_int_local(data, "row_ptr");
    ctx->col_idx = get_tensor_int_local(data, "col_idx");

    if (!ctx->row_ptr || !ctx->col_idx) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        free(ctx);
        return NULL;
    }
    ctx->comp_out = (int*)calloc((size_t)(ctx->n), sizeof(int));
    return ctx;
}

void task_run(void* test_data) {
    CCTaskContext* ctx = (CCTaskContext*)test_data;
    solution_compute(ctx->n, ctx->neighbor_rounds, ctx->num_samples, ctx->row_ptr, ctx->col_idx, ctx->comp_out);
}

void task_write_output(void* test_data, const char* output_path) {
    CCTaskContext* ctx = (CCTaskContext*)test_data;
    FILE *f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->n; ++i) {
        fprintf(f, "%d\n", ctx->comp_out[i]);
    }
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    CCTaskContext* ctx = (CCTaskContext*)test_data;
    free(ctx->comp_out);
    free(ctx);
}
