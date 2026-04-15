#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void solution_init(int n,
                          int neighbor_rounds,
                          int num_samples,
                          const int *row_ptr,
                          const int *col_idx);
extern void solution_compute(int *comp_out);
extern void solution_free(void);

typedef struct {
    int n;
    int *comp_out;
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
    int n = (int)get_param(data, "n");
    int neighbor_rounds = (int)get_param(data, "neighbor_rounds");
    int num_samples = (int)get_param(data, "num_samples");
    const int *row_ptr = get_tensor_int_local(data, "row_ptr");
    const int *col_idx = get_tensor_int_local(data, "col_idx");
    if (!row_ptr || !col_idx) {
        fprintf(stderr, "[task_io] Missing GAPBS CC input tensor\n");
        return NULL;
    }
    solution_init(n, neighbor_rounds, num_samples, row_ptr, col_idx);
    CCTaskContext *ctx = (CCTaskContext*)calloc(1, sizeof(CCTaskContext));
    if (!ctx) return NULL;
    ctx->n = n;
    ctx->comp_out = (int*)malloc((size_t)n * sizeof(int));
    if (!ctx->comp_out) {
        free(ctx);
        return NULL;
    }
    return ctx;
}

void task_run(void* test_data) {
    CCTaskContext *ctx = (CCTaskContext*)test_data;
    solution_compute(ctx->comp_out);
}

void task_write_output(void* test_data, const char* output_path) {
    CCTaskContext *ctx = (CCTaskContext*)test_data;
    FILE *f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->n; ++i) {
        fprintf(f, "%d\n", ctx->comp_out[i]);
    }
    fclose(f);
}

void task_cleanup(void* test_data) {
    CCTaskContext *ctx = (CCTaskContext*)test_data;
    if (!ctx) return;
    solution_free();
    free(ctx->comp_out);
    free(ctx);
}
