#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif
extern void solution_init(int num_nodes, int num_edges, int source,
                          const int* node_start,
                          const int* node_degree,
                          const int* edge_dst);
extern void solution_compute(int* out_dist);
extern void solution_free(void);
#ifdef __cplusplus
}
#endif

typedef struct {
    int n;
    int* out_dist;
} TaskIOContext;

#ifdef __cplusplus
extern "C" {
#endif

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    int num_nodes = (int)get_param(data, "num_nodes");
    int num_edges = (int)get_param(data, "num_edges");
    int source = (int)get_param(data, "source");
    const int* node_start = get_tensor_int(data, "node_start");
    const int* node_degree = get_tensor_int(data, "node_degree");
    const int* edge_dst = get_tensor_int(data, "edge_dst");
    if (!node_start || !node_degree || !edge_dst) {
        fprintf(stderr, "[task_io] Missing graph tensors\n");
        return NULL;
    }
    solution_init(num_nodes, num_edges, source, node_start, node_degree, edge_dst);
    TaskIOContext* ctx = (TaskIOContext*)calloc(1, sizeof(TaskIOContext));
    ctx->n = num_nodes;
    ctx->out_dist = (int*)calloc((size_t)num_nodes, sizeof(int));
    return ctx;
}

void task_run(void* test_data) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_compute(ctx->out_dist);
}

void task_write_output(void* test_data, const char* output_path) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->n; ++i) fprintf(f, "%d\n", ctx->out_dist[i]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_free();
    free(ctx->out_dist);
    free(ctx);
}

#ifdef __cplusplus
}
#endif
