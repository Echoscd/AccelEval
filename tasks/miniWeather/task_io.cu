// task_io.cu — miniWeather GPU I/O adapter (compute_only interface)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define MW_NUM_OUTPUTS 6

#ifdef __cplusplus
extern "C" {
#endif

extern void solution_compute(
    int nx_in, int nz_in,
    int sim_time_in, int data_spec_in,
    double *output  // [6]: d_mass, d_te, L2_dens, L2_umom, L2_wmom, L2_rhot
);
extern void solution_free(void);
// Weak default: LLM does not need to implement solution_free
extern "C" __attribute__((weak)) void solution_free(void) { }

#ifdef __cplusplus
}
#endif

typedef struct {
    int NX, NZ, SIM_TIME, DATA_SPEC;
    double output[MW_NUM_OUTPUTS];
} MWContext;

#ifdef __cplusplus
extern "C" {
#endif

void* task_setup(const TaskData* data, const char* data_dir) {
    MWContext* ctx = (MWContext*)calloc(1, sizeof(MWContext));
    ctx->NX        = (int)get_param(data, "NX");
    ctx->NZ        = (int)get_param(data, "NZ");
    ctx->SIM_TIME  = (int)get_param(data, "SIM_TIME");
    ctx->DATA_SPEC = (int)get_param(data, "DATA_SPEC");
    return ctx;
}

void task_run(void* test_data) {
    MWContext* ctx = (MWContext*)test_data;
    solution_compute(ctx->NX, ctx->NZ, ctx->SIM_TIME, ctx->DATA_SPEC, ctx->output);
}

void task_write_output(void* test_data, const char* output_path) {
    MWContext* ctx = (MWContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < MW_NUM_OUTPUTS; i++)
        fprintf(f, "%.15e\n", ctx->output[i]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    MWContext* ctx = (MWContext*)test_data;
    solution_free();
    free(ctx);
}

#ifdef __cplusplus
}
#endif
