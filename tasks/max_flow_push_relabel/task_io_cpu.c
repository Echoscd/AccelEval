// task_io_cpu.c -- max_flow_push_relabel CPU I/O adapter

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>

extern void solution_init(int num_nodes, int num_arcs,
                          const int* tails, const int* heads, const int* caps,
                          int source, int sink);
extern void solution_compute(int num_nodes, int* max_flow_out);
extern void solution_free(void);

// Weak default: LLM does not need to implement solution_free
__attribute__((weak)) void solution_free(void) { }

typedef struct {
    int  num_nodes;
    int  max_flow;
} MFContext;

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    int num_nodes = (int)get_param(data, "num_nodes");
    int num_arcs  = (int)get_param(data, "num_arcs");
    int source    = (int)get_param(data, "source");
    int sink      = (int)get_param(data, "sink");

    const int* tails = get_tensor_int(data, "tails");
    const int* heads = get_tensor_int(data, "heads");
    const int* caps  = get_tensor_int(data, "caps");
    if (!tails || !heads || !caps) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        return NULL;
    }

    MFContext* ctx = (MFContext*)calloc(1, sizeof(MFContext));
    ctx->num_nodes = num_nodes;

    solution_init(num_nodes, num_arcs, tails, heads, caps, source, sink);
    return ctx;
}

void task_run(void* test_data) {
    MFContext* ctx = (MFContext*)test_data;
    solution_compute(ctx->num_nodes, &ctx->max_flow);
}

void task_write_output(void* test_data, const char* output_path) {
    MFContext* ctx = (MFContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    fprintf(f, "%d\n", ctx->max_flow);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    MFContext* ctx = (MFContext*)test_data;
    solution_free();
    free(ctx);
}
