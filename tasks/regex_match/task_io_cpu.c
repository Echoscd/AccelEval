// task_io_cpu.c — unified compute_only interface (auto-migrated)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void solution_compute(int num_states,
                             int num_symbols,
                             int start_state,
                             int num_strings,
                             int total_chars,
                             const int* trans_offsets,
                             const int* trans_targets,
                             const int* eps_offsets,
                             const int* eps_targets,
                             const int* is_accept,
                             const int* str_offsets,
                             const int* str_data,
                             int* results);

typedef struct {
    int num_states;
    int num_symbols;
    int start_state;
    int num_strings;
    int total_chars;
    const int* trans_offsets;
    const int* trans_targets;
    const int* eps_offsets;
    const int* eps_targets;
    const int* is_accept;
    const int* str_offsets;
    const int* str_data;
    int* results;
} TaskIOContext;

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    TaskIOContext* ctx = (TaskIOContext*)calloc(1, sizeof(TaskIOContext));
    if (!ctx) return NULL;
    ctx->num_states = (int)get_param(data, "num_states");
    ctx->num_symbols = (int)get_param(data, "num_symbols");
    ctx->start_state = (int)get_param(data, "start_state");
    ctx->num_strings = (int)get_param(data, "num_strings");
    ctx->total_chars = (int)get_param(data, "total_chars");
    ctx->trans_offsets = get_tensor_int(data, "trans_offsets");
    ctx->trans_targets = get_tensor_int(data, "trans_targets");
    ctx->eps_offsets = get_tensor_int(data, "eps_offsets");
    ctx->eps_targets = get_tensor_int(data, "eps_targets");
    ctx->is_accept = get_tensor_int(data, "is_accept");
    ctx->str_offsets = get_tensor_int(data, "str_offsets");
    ctx->str_data = get_tensor_int(data, "str_data");

    if (!ctx->trans_offsets || !ctx->trans_targets || !ctx->eps_offsets || !ctx->eps_targets || !ctx->is_accept || !ctx->str_offsets || !ctx->str_data) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        free(ctx);
        return NULL;
    }
    ctx->results = (int*)calloc((size_t)(ctx->num_strings), sizeof(int));
    return ctx;
}

void task_run(void* test_data) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_compute(ctx->num_states, ctx->num_symbols, ctx->start_state, ctx->num_strings, ctx->total_chars, ctx->trans_offsets, ctx->trans_targets, ctx->eps_offsets, ctx->eps_targets, ctx->is_accept, ctx->str_offsets, ctx->str_data, ctx->results);
}

void task_write_output(void* test_data, const char* output_path) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->num_strings; i++) {
        fprintf(f, "%d\n", ctx->results[i]);
    }
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    free(ctx->results);
    free(ctx);
}
