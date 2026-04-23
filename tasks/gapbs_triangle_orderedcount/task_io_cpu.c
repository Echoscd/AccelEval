// task_io_cpu.c — unified compute_only interface (auto-migrated)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void solution_compute(int n,
                             const int * row_ptr,
                             const int * col_idx,
                             unsigned long long * triangle_count_out);

typedef struct {
    int n;
    const int * row_ptr;
    const int * col_idx;
    unsigned long long * triangle_count_out;
} TCTaskContext;

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
    TCTaskContext* ctx = (TCTaskContext*)calloc(1, sizeof(TCTaskContext));
    if (!ctx) return NULL;
    ctx->n = (int)get_param(data, "n");
    ctx->row_ptr = get_tensor_int_local(data, "row_ptr");
    ctx->col_idx = get_tensor_int_local(data, "col_idx");

    if (!ctx->row_ptr || !ctx->col_idx) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        free(ctx);
        return NULL;
    }
    ctx->triangle_count_out = (unsigned long long*)calloc((size_t)(1), sizeof(unsigned long long));
    return ctx;
}

void task_run(void* test_data) {
    TCTaskContext* ctx = (TCTaskContext*)test_data;
    solution_compute(ctx->n, ctx->row_ptr, ctx->col_idx, ctx->triangle_count_out);
}

void task_write_output(void* test_data, const char* output_path) {
    TCTaskContext* ctx = (TCTaskContext*)test_data;
    FILE *f = fopen(output_path, "w");
    if (!f) return;
    fprintf(f, "%llu\n", (unsigned long long)ctx->triangle_count_out[0]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    TCTaskContext* ctx = (TCTaskContext*)test_data;
    free(ctx->triangle_count_out);
    free(ctx);
}
