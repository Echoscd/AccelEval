#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void solution_init(int n, const int *row_ptr, const int *col_idx);
extern "C" void solution_compute(unsigned long long *triangle_count_out);
extern "C" void solution_free(void);

typedef struct {
    unsigned long long triangle_count;
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

extern "C" void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    int n = (int)get_param(data, "n");
    const int *row_ptr = get_tensor_int_local(data, "row_ptr");
    const int *col_idx = get_tensor_int_local(data, "col_idx");
    if (!row_ptr || !col_idx) {
        fprintf(stderr, "[task_io] Missing GAPBS Triangle Count input tensor\n");
        return NULL;
    }
    solution_init(n, row_ptr, col_idx);
    TCTaskContext *ctx = (TCTaskContext*)calloc(1, sizeof(TCTaskContext));
    return ctx;
}

extern "C" void task_run(void* test_data) {
    TCTaskContext *ctx = (TCTaskContext*)test_data;
    solution_compute(&ctx->triangle_count);
}

extern "C" void task_write_output(void* test_data, const char* output_path) {
    TCTaskContext *ctx = (TCTaskContext*)test_data;
    FILE *f = fopen(output_path, "w");
    if (!f) return;
    fprintf(f, "%llu\n", (unsigned long long)ctx->triangle_count);
    fclose(f);
}

extern "C" void task_cleanup(void* test_data) {
    TCTaskContext *ctx = (TCTaskContext*)test_data;
    if (!ctx) return;
    solution_free();
    free(ctx);
}
