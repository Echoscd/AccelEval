// task_io.cu — unified compute_only interface (auto-migrated)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

extern "C" void solution_compute(int num_entries,
                             int num_features,
                             const float* subjects,
                             const float* query,
                             float* distances);

typedef struct {
    int num_entries;
    int num_features;
    const float* subjects;
    const float* query;
    float* distances;
} DTWContext;

extern "C" void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    DTWContext* ctx = (DTWContext*)calloc(1, sizeof(DTWContext));
    if (!ctx) return NULL;
    ctx->num_entries = (int)get_param(data, "num_entries");
    ctx->num_features = (int)get_param(data, "num_features");
    ctx->subjects = get_tensor_float(data, "subjects");
    ctx->query = get_tensor_float(data, "query");

    if (!ctx->subjects || !ctx->query) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        free(ctx);
        return NULL;
    }
    ctx->distances = (float*)calloc((size_t)(ctx->num_entries), sizeof(float));
    return ctx;
}

extern "C" void task_run(void* test_data) {
    DTWContext* ctx = (DTWContext*)test_data;
    solution_compute(ctx->num_entries, ctx->num_features, ctx->subjects, ctx->query, ctx->distances);
}

extern "C" void task_write_output(void* test_data, const char* output_path) {
    DTWContext* ctx = (DTWContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->num_entries; i++)
        fprintf(f, "%.6e\n", ctx->distances[i]);
    fclose(f);
}

extern "C" void task_cleanup(void* test_data) {
    if (!test_data) return;
    DTWContext* ctx = (DTWContext*)test_data;
    free(ctx->distances);
    free(ctx);
}
