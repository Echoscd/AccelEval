// task_io.cu — bellman_ford unified compute_only interface

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

extern void solution_compute(int V, int E,
                             const int* h_row_offsets,
                             const int* h_col_indices,
                             const float* h_weights,
                             int num_requests,
                             const int* h_sources,
                             const int* h_targets,
                             float* h_distances);

#ifdef __cplusplus
}
#endif

typedef struct {
    int V;
    int E;
    const int* h_row_offsets;
    const int* h_col_indices;
    const float* h_weights;
    int num_requests;
    int* h_sources;
    int* h_targets;
    float* h_distances;
} TaskIOContext;

#ifdef __cplusplus
extern "C" {
#endif

void* task_setup(const TaskData* data, const char* data_dir) {
    TaskIOContext* ctx = (TaskIOContext*)calloc(1, sizeof(TaskIOContext));
    if (!ctx) return NULL;

    ctx->V = (int)get_param(data, "V");
    ctx->E = (int)get_param(data, "E");
    ctx->h_row_offsets = get_tensor_int(data, "row_offsets");
    ctx->h_col_indices = get_tensor_int(data, "col_indices");
    ctx->h_weights = get_tensor_float(data, "weights");
    if (!ctx->h_row_offsets || !ctx->h_col_indices || !ctx->h_weights) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        free(ctx);
        return NULL;
    }

    char path[512];
    snprintf(path, sizeof(path), "%s/requests.txt", data_dir);
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[task_io] Missing requests.txt\n");
        free(ctx);
        return NULL;
    }
    char line[256];
    int n = 0;
    while (fgets(line, sizeof(line), f)) {
        int s, t;
        if (sscanf(line, "%d %d", &s, &t) == 2) n++;
    }
    rewind(f);

    ctx->num_requests = n;
    ctx->h_sources   = (int*)calloc((size_t)n, sizeof(int));
    ctx->h_targets   = (int*)calloc((size_t)n, sizeof(int));
    ctx->h_distances = (float*)calloc((size_t)n, sizeof(float));

    int idx = 0;
    while (fgets(line, sizeof(line), f)) {
        int s, t;
        if (sscanf(line, "%d %d", &s, &t) == 2) {
            ctx->h_sources[idx] = s;
            ctx->h_targets[idx] = t;
            idx++;
        }
    }
    fclose(f);
    return ctx;
}

void task_run(void* test_data) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_compute(ctx->V, ctx->E,
                     ctx->h_row_offsets, ctx->h_col_indices, ctx->h_weights,
                     ctx->num_requests, ctx->h_sources, ctx->h_targets,
                     ctx->h_distances);
}

void task_write_output(void* test_data, const char* output_path) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->num_requests; i++)
        fprintf(f, "%.6e\n", ctx->h_distances[i]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    free(ctx->h_sources);
    free(ctx->h_targets);
    free(ctx->h_distances);
    free(ctx);
}

#ifdef __cplusplus
}
#endif
