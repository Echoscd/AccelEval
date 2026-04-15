// task_io.cu -- pdlp GPU I/O adapter

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

extern void solution_init(int num_vars, int num_constraints, int nnz,
                          int num_iters,
                          const float* obj, const float* var_lb,
                          const float* var_ub, const float* con_lb,
                          const float* con_ub, const int* col_ptrs,
                          const int* row_indices, const float* values,
                          float step_size, float primal_weight);
extern void solution_compute(int num_vars, int num_constraints, float* primal_out);
extern void solution_free(void);
// Weak default: LLM does not need to implement solution_free
extern "C" __attribute__((weak)) void solution_free(void) { }

#ifdef __cplusplus
}
#endif

typedef struct {
    int    num_vars;
    int    num_constraints;
    float* primal_out;
} PDLPContext;

#ifdef __cplusplus
extern "C" {
#endif

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    int num_vars        = (int)get_param(data, "num_vars");
    int num_constraints = (int)get_param(data, "num_constraints");
    int nnz             = (int)get_param(data, "nnz");
    int num_iters       = (int)get_param(data, "num_iters");

    const float* obj         = get_tensor_float(data, "obj");
    const float* var_lb      = get_tensor_float(data, "var_lb");
    const float* var_ub      = get_tensor_float(data, "var_ub");
    const float* con_lb      = get_tensor_float(data, "con_lb");
    const float* con_ub      = get_tensor_float(data, "con_ub");
    const int*   col_ptrs    = get_tensor_int(data, "col_ptrs");
    const int*   row_indices = get_tensor_int(data, "row_indices");
    const float* values      = get_tensor_float(data, "values");
    const float* step_size_arr     = get_tensor_float(data, "step_size");
    const float* primal_weight_arr = get_tensor_float(data, "primal_weight");

    if (!obj || !var_lb || !var_ub || !con_lb || !con_ub ||
        !col_ptrs || !row_indices || !values ||
        !step_size_arr || !primal_weight_arr) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        return NULL;
    }

    float step_size    = step_size_arr[0];
    float primal_weight = primal_weight_arr[0];

    PDLPContext* ctx = (PDLPContext*)calloc(1, sizeof(PDLPContext));
    ctx->num_vars        = num_vars;
    ctx->num_constraints = num_constraints;
    ctx->primal_out      = (float*)calloc(num_vars, sizeof(float));

    solution_init(num_vars, num_constraints, nnz, num_iters,
                  obj, var_lb, var_ub, con_lb, con_ub,
                  col_ptrs, row_indices, values,
                  step_size, primal_weight);
    return ctx;
}

void task_run(void* test_data) {
    PDLPContext* ctx = (PDLPContext*)test_data;
    solution_compute(ctx->num_vars, ctx->num_constraints, ctx->primal_out);
}

void task_write_output(void* test_data, const char* output_path) {
    PDLPContext* ctx = (PDLPContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->num_vars; i++)
        fprintf(f, "%.6e\n", ctx->primal_out[i]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    PDLPContext* ctx = (PDLPContext*)test_data;
    solution_free();
    free(ctx->primal_out);
    free(ctx);
}

#ifdef __cplusplus
}
#endif
