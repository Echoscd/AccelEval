#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void solution_init(int n, int m, const float *A, const float *p, const float *r);
extern "C" void solution_compute(float *s_out, float *q_out);
extern "C" void solution_free(void);

typedef struct {
    int n, m;
    float *s_out;
    float *q_out;
} BiCGTaskContext;

static float* get_tensor_float_local(const TaskData* data, const char* name) {
    if (!data || !name) return NULL;
    for (int i = 0; i < data->num_inputs; ++i) {
        if (strcmp(data->inputs[i].name, name) == 0) {
            if (data->inputs[i].dtype != 1) return NULL;
            return (float*)data->inputs[i].data;
        }
    }
    return NULL;
}

extern "C" void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    int n = (int)get_param(data, "n");
    int m = (int)get_param(data, "m");
    const float *A = get_tensor_float_local(data, "A");
    const float *p = get_tensor_float_local(data, "p");
    const float *r = get_tensor_float_local(data, "r");
    if (!A || !p || !r) {
        fprintf(stderr, "[task_io] Missing BiCG input tensor\n");
        return NULL;
    }
    solution_init(n, m, A, p, r);
    BiCGTaskContext *ctx = (BiCGTaskContext*)calloc(1, sizeof(BiCGTaskContext));
    if (!ctx) return NULL;
    ctx->n = n;
    ctx->m = m;
    ctx->s_out = (float*)malloc((size_t)m * sizeof(float));
    ctx->q_out = (float*)malloc((size_t)n * sizeof(float));
    if (!ctx->s_out || !ctx->q_out) {
        free(ctx->s_out);
        free(ctx->q_out);
        free(ctx);
        return NULL;
    }
    return ctx;
}

extern "C" void task_run(void* test_data) {
    BiCGTaskContext *ctx = (BiCGTaskContext*)test_data;
    solution_compute(ctx->s_out, ctx->q_out);
}

extern "C" void task_write_output(void* test_data, const char* output_path) {
    BiCGTaskContext *ctx = (BiCGTaskContext*)test_data;
    FILE *f = fopen(output_path, "w");
    if (!f) return;
    for (int j = 0; j < ctx->m; ++j) {
        fprintf(f, "%.8e\n", ctx->s_out[j]);
    }
    for (int i = 0; i < ctx->n; ++i) {
        fprintf(f, "%.8e\n", ctx->q_out[i]);
    }
    fclose(f);
}

extern "C" void task_cleanup(void* test_data) {
    BiCGTaskContext *ctx = (BiCGTaskContext*)test_data;
    if (!ctx) return;
    solution_free();
    free(ctx->s_out);
    free(ctx->q_out);
    free(ctx);
}
