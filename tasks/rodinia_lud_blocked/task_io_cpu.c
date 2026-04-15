#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>

extern void solution_init(int n, const float* A0);
extern void solution_compute(float* LU);
extern void solution_free(void);

// Weak default: LLM does not need to implement solution_free
__attribute__((weak)) void solution_free(void) { }

typedef struct {
    int n;
    float* LU;
} TaskIOContext;

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    int n = (int)get_param(data, "n");
    const float* A0 = get_tensor_float(data, "A0");
    if (!A0) {
        fprintf(stderr, "[task_io] Missing tensor A0\n");
        return NULL;
    }
    solution_init(n, A0);
    TaskIOContext* ctx = (TaskIOContext*)calloc(1, sizeof(TaskIOContext));
    ctx->n = n;
    ctx->LU = (float*)malloc((size_t)n * (size_t)n * sizeof(float));
    if (!ctx->LU) {
        free(ctx);
        return NULL;
    }
    return ctx;
}

void task_run(void* test_data) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_compute(ctx->LU);
}

void task_write_output(void* test_data, const char* output_path) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    int total = ctx->n * ctx->n;
    for (int i = 0; i < total; i++) {
        fprintf(f, "%.8f\n", ctx->LU[i]);
    }
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_free();
    free(ctx->LU);
    free(ctx);
}
