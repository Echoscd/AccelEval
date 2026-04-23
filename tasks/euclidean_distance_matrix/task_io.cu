// task_io.cu — unified compute_only interface (auto-migrated)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

extern "C" void solution_compute(int ref_nb,
                             int query_nb,
                             int dim,
                             const float* ref,
                             const float* query,
                             float* dist);

typedef struct {
    int ref_nb;
    int query_nb;
    int dim;
    const float* ref;
    const float* query;
    float* dist;
} EDContext;

extern "C" void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    EDContext* ctx = (EDContext*)calloc(1, sizeof(EDContext));
    if (!ctx) return NULL;
    ctx->ref_nb = (int)get_param(data, "ref_nb");
    ctx->query_nb = (int)get_param(data, "query_nb");
    ctx->dim = (int)get_param(data, "dim");
    ctx->ref = get_tensor_float(data, "ref");
    ctx->query = get_tensor_float(data, "query");

    if (!ctx->ref || !ctx->query) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        free(ctx);
        return NULL;
    }
    ctx->dist = (float*)calloc((size_t)(ctx->query_nb * ctx->ref_nb), sizeof(float));
    return ctx;
}

extern "C" void task_run(void* test_data) {
    EDContext* ctx = (EDContext*)test_data;
    solution_compute(ctx->ref_nb, ctx->query_nb, ctx->dim, ctx->ref, ctx->query, ctx->dist);
}

extern "C" void task_write_output(void* test_data, const char* output_path) {
    EDContext* ctx = (EDContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    size_t total = (size_t)ctx->query_nb * ctx->ref_nb;
    for (size_t i = 0; i < total; i++)
        fprintf(f, "%.6e\n", ctx->dist[i]);
    fclose(f);
}

extern "C" void task_cleanup(void* test_data) {
    if (!test_data) return;
    EDContext* ctx = (EDContext*)test_data;
    free(ctx->dist);
    free(ctx);
}
