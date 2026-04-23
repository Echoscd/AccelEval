// task_io_cpu.c — unified compute_only interface (auto-migrated)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void solution_compute(int num_nodes,
                             int num_arcs,
                             const int* tails,
                             const int* heads,
                             const int* caps,
                             int source,
                             int sink,
                             int* max_flow_out);

typedef struct {
    int num_nodes;
    int num_arcs;
    const int* tails;
    const int* heads;
    const int* caps;
    int source;
    int sink;
    int* max_flow_out;
} MFContext;

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    MFContext* ctx = (MFContext*)calloc(1, sizeof(MFContext));
    if (!ctx) return NULL;
    ctx->num_nodes = (int)get_param(data, "num_nodes");
    ctx->num_arcs = (int)get_param(data, "num_arcs");
    ctx->tails = get_tensor_int(data, "tails");
    ctx->heads = get_tensor_int(data, "heads");
    ctx->caps = get_tensor_int(data, "caps");
    ctx->source = (int)get_param(data, "source");
    ctx->sink = (int)get_param(data, "sink");

    if (!ctx->tails || !ctx->heads || !ctx->caps) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        free(ctx);
        return NULL;
    }
    ctx->max_flow_out = (int*)calloc((size_t)(1), sizeof(int));
    return ctx;
}

void task_run(void* test_data) {
    MFContext* ctx = (MFContext*)test_data;
    solution_compute(ctx->num_nodes, ctx->num_arcs, ctx->tails, ctx->heads, ctx->caps, ctx->source, ctx->sink, ctx->max_flow_out);
}

void task_write_output(void* test_data, const char* output_path) {
    MFContext* ctx = (MFContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    fprintf(f, "%d\n", ctx->max_flow_out[0]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    MFContext* ctx = (MFContext*)test_data;
    free(ctx->max_flow_out);
    free(ctx);
}
