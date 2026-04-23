// task_io_cpu.c — nash_flows_over_time unified compute_only interface (CPU)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void solution_compute(
    int num_nodes, int num_edges, int num_steps,
    const int* edge_u, const int* edge_v,
    const float* edge_capacity, const int* edge_transit_time,
    int num_requests, const float* inflow_rates, float* results);

typedef struct {
    int num_nodes;
    int num_edges;
    int num_steps;
    const int* edge_u;
    const int* edge_v;
    const float* edge_capacity;
    const int* edge_transit_time;
    int num_requests;
    float* inflow_rates;
    float* results;
} Ctx;

void* task_setup(const TaskData* data, const char* data_dir) {
    Ctx* ctx = (Ctx*)calloc(1, sizeof(Ctx));
    if (!ctx) return NULL;
    ctx->num_nodes = (int)get_param(data, "num_nodes");
    ctx->num_edges = (int)get_param(data, "num_edges");
    ctx->num_steps = (int)get_param(data, "num_steps");
    ctx->edge_u = get_tensor_int(data, "edge_u");
    ctx->edge_v = get_tensor_int(data, "edge_v");
    ctx->edge_capacity = get_tensor_float(data, "edge_capacity");
    ctx->edge_transit_time = get_tensor_int(data, "edge_transit_time");
    if (!ctx->edge_u || !ctx->edge_v || !ctx->edge_capacity || !ctx->edge_transit_time) {
        fprintf(stderr, "[task_io] Missing tensor data\n"); free(ctx); return NULL;
    }

    char req_path[512];
    snprintf(req_path, sizeof(req_path), "%s/requests.txt", data_dir);
    FILE* f = fopen(req_path, "r");
    if (!f) { fprintf(stderr, "[task_io] Missing requests.txt\n"); free(ctx); return NULL; }
    float rates[1024];
    int n = 0;
    while (n < 1024 && fscanf(f, "%f", &rates[n]) == 1) n++;
    fclose(f);

    ctx->num_requests = n;
    ctx->inflow_rates = (float*)malloc((size_t)n * sizeof(float));
    memcpy(ctx->inflow_rates, rates, (size_t)n * sizeof(float));
    ctx->results = (float*)calloc((size_t)n, sizeof(float));
    return ctx;
}

void task_run(void* test_data) {
    Ctx* ctx = (Ctx*)test_data;
    solution_compute(ctx->num_nodes, ctx->num_edges, ctx->num_steps,
                     ctx->edge_u, ctx->edge_v,
                     ctx->edge_capacity, ctx->edge_transit_time,
                     ctx->num_requests, ctx->inflow_rates, ctx->results);
}

void task_write_output(void* test_data, const char* output_path) {
    Ctx* ctx = (Ctx*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->num_requests; i++)
        fprintf(f, "%.6f\n", ctx->results[i]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    Ctx* ctx = (Ctx*)test_data;
    free(ctx->inflow_rates); free(ctx->results); free(ctx);
}
