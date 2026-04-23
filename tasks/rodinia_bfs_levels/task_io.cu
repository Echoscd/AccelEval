// task_io.cu — unified compute_only interface (auto-migrated)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

extern "C" void solution_compute(int num_nodes,
                             int num_edges,
                             int source,
                             const int* node_start,
                             const int* node_degree,
                             const int* edge_dst,
                             int* out_dist);

typedef struct {
    int num_nodes;
    int num_edges;
    int source;
    const int* node_start;
    const int* node_degree;
    const int* edge_dst;
    int n;
    int* out_dist;
} TaskIOContext;

extern "C" void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    TaskIOContext* ctx = (TaskIOContext*)calloc(1, sizeof(TaskIOContext));
    if (!ctx) return NULL;
    ctx->num_nodes = (int)get_param(data, "num_nodes");
    ctx->num_edges = (int)get_param(data, "num_edges");
    ctx->source = (int)get_param(data, "source");
    ctx->node_start = get_tensor_int(data, "node_start");
    ctx->node_degree = get_tensor_int(data, "node_degree");
    ctx->edge_dst = get_tensor_int(data, "edge_dst");

    if (!ctx->node_start || !ctx->node_degree || !ctx->edge_dst) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        free(ctx);
        return NULL;
    }
    ctx->n = ctx->num_nodes;
    ctx->out_dist = (int*)calloc((size_t)(ctx->num_nodes), sizeof(int));
    return ctx;
}

extern "C" void task_run(void* test_data) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_compute(ctx->num_nodes, ctx->num_edges, ctx->source, ctx->node_start, ctx->node_degree, ctx->edge_dst, ctx->out_dist);
}

extern "C" void task_write_output(void* test_data, const char* output_path) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->n; ++i) fprintf(f, "%d\n", ctx->out_dist[i]);
    fclose(f);
}

extern "C" void task_cleanup(void* test_data) {
    if (!test_data) return;
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    free(ctx->out_dist);
    free(ctx);
}
