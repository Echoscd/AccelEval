#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void solution_init(int n,
                               int max_iters,
                               const int *in_row_ptr,
                               const int *in_col_idx,
                               const int *out_degree);
extern "C" void solution_compute(float *scores_out);
extern "C" void solution_free(void);

typedef struct {
    int n;
    float *scores_out;
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

extern "C" void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    int n = (int)get_param(data, "n");
    int max_iters = (int)get_param(data, "max_iters");
    const int *in_row_ptr = get_tensor_int_local(data, "in_row_ptr");
    const int *in_col_idx = get_tensor_int_local(data, "in_col_idx");
    const int *out_degree = get_tensor_int_local(data, "out_degree");
    if (!in_row_ptr || !in_col_idx || !out_degree) {
        fprintf(stderr, "[task_io] Missing GAPBS PageRank input tensor\n");
        return NULL;
    }
    solution_init(n, max_iters, in_row_ptr, in_col_idx, out_degree);
    PRTaskContext *ctx = (PRTaskContext*)calloc(1, sizeof(PRTaskContext));
    if (!ctx) return NULL;
    ctx->n = n;
    ctx->scores_out = (float*)malloc((size_t)n * sizeof(float));
    if (!ctx->scores_out) {
        free(ctx);
        return NULL;
    }
    return ctx;
}

extern "C" void task_run(void* test_data) {
    PRTaskContext *ctx = (PRTaskContext*)test_data;
    solution_compute(ctx->scores_out);
}

extern "C" void task_write_output(void* test_data, const char* output_path) {
    PRTaskContext *ctx = (PRTaskContext*)test_data;
    FILE *f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->n; ++i) {
        fprintf(f, "%.8e\n", ctx->scores_out[i]);
    }
    fclose(f);
}

extern "C" void task_cleanup(void* test_data) {
    PRTaskContext *ctx = (PRTaskContext*)test_data;
    if (!ctx) return;
    solution_free();
    free(ctx->scores_out);
    free(ctx);
}
