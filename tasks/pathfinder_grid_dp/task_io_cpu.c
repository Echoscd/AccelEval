// task_io_cpu.c -- pathfinder_grid_dp CPU I/O adapter

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>

extern void solution_init(int rows, int cols, const int* wall);
extern void solution_compute(int* out_costs);
extern void solution_free(void);

typedef struct {
    int cols;
    int* out_costs;
} TaskIOContext;

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    int rows = (int)get_param(data, "rows");
    int cols = (int)get_param(data, "cols");
    const int* wall = get_tensor_int(data, "wall");
    if (!wall) {
        fprintf(stderr, "[task_io_cpu] Missing tensor 'wall'\n");
        return NULL;
    }

    solution_init(rows, cols, wall);

    TaskIOContext* ctx = (TaskIOContext*)calloc(1, sizeof(TaskIOContext));
    ctx->cols = cols;
    ctx->out_costs = (int*)calloc((size_t)cols, sizeof(int));
    return ctx;
}

void task_run(void* test_data) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_compute(ctx->out_costs);
}

void task_write_output(void* test_data, const char* output_path) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int c = 0; c < ctx->cols; ++c) {
        fprintf(f, "%d\n", ctx->out_costs[c]);
    }
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_free();
    free(ctx->out_costs);
    free(ctx);
}
