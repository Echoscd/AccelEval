// task_io.cu — unified compute_only interface (auto-migrated)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

extern "C" void solution_compute(int num_vars,
                             int num_constraints,
                             int nnz,
                             int num_iters,
                             const float* obj,
                             const float* var_lb,
                             const float* var_ub,
                             const float* con_lb,
                             const float* con_ub,
                             const int* col_ptrs,
                             const int* row_indices,
                             const float* values,
                             float step_size,
                             float primal_weight,
                             float* primal_out);

typedef struct {
    int num_vars;
    int num_constraints;
    int nnz;
    int num_iters;
    const float* obj;
    const float* var_lb;
    const float* var_ub;
    const float* con_lb;
    const float* con_ub;
    const int* col_ptrs;
    const int* row_indices;
    const float* values;
    float step_size;
    float primal_weight;
    float* primal_out;
} PDLPContext;

extern "C" void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    PDLPContext* ctx = (PDLPContext*)calloc(1, sizeof(PDLPContext));
    if (!ctx) return NULL;
    const float* step_size_arr     = get_tensor_float(data, "step_size");
    const float* primal_weight_arr = get_tensor_float(data, "primal_weight");
    ctx->num_vars = (int)get_param(data, "num_vars");
    ctx->num_constraints = (int)get_param(data, "num_constraints");
    ctx->nnz = (int)get_param(data, "nnz");
    ctx->num_iters = (int)get_param(data, "num_iters");
    ctx->obj = get_tensor_float(data, "obj");
    ctx->var_lb = get_tensor_float(data, "var_lb");
    ctx->var_ub = get_tensor_float(data, "var_ub");
    ctx->con_lb = get_tensor_float(data, "con_lb");
    ctx->con_ub = get_tensor_float(data, "con_ub");
    ctx->col_ptrs = get_tensor_int(data, "col_ptrs");
    ctx->row_indices = get_tensor_int(data, "row_indices");
    ctx->values = get_tensor_float(data, "values");
    ctx->step_size = step_size_arr[0];
    ctx->primal_weight = primal_weight_arr[0];

    if (!ctx->obj || !ctx->var_lb || !ctx->var_ub || !ctx->con_lb || !ctx->con_ub || !ctx->col_ptrs || !ctx->row_indices || !ctx->values) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        free(ctx);
        return NULL;
    }
    ctx->primal_out = (float*)calloc((size_t)(ctx->num_vars), sizeof(float));
    return ctx;
}

extern "C" void task_run(void* test_data) {
    PDLPContext* ctx = (PDLPContext*)test_data;
    solution_compute(ctx->num_vars, ctx->num_constraints, ctx->nnz, ctx->num_iters, ctx->obj, ctx->var_lb, ctx->var_ub, ctx->con_lb, ctx->con_ub, ctx->col_ptrs, ctx->row_indices, ctx->values, ctx->step_size, ctx->primal_weight, ctx->primal_out);
}

extern "C" void task_write_output(void* test_data, const char* output_path) {
    PDLPContext* ctx = (PDLPContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->num_vars; i++)
        fprintf(f, "%.6e\n", ctx->primal_out[i]);
    fclose(f);
}

extern "C" void task_cleanup(void* test_data) {
    if (!test_data) return;
    PDLPContext* ctx = (PDLPContext*)test_data;
    free(ctx->primal_out);
    free(ctx);
}
