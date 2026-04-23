// task_io_cpu.c — unified compute_only interface (auto-migrated)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void solution_compute(int N,
                             int num_steps,
                             float risk_free,
                             float volatility,
                             float strike,
                             float spot,
                             float time_to_maturity,
                             unsigned int base_seed,
                             float* payoffs);

typedef struct {
    int N;
    int num_steps;
    float risk_free;
    float volatility;
    float strike;
    float spot;
    float time_to_maturity;
    unsigned int base_seed;
    float* payoffs;
} MCContext;

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    MCContext* ctx = (MCContext*)calloc(1, sizeof(MCContext));
    if (!ctx) return NULL;
    int risk_free_x10000   = (int)get_param(data, "risk_free_x10000");
    int volatility_x10000  = (int)get_param(data, "volatility_x10000");
    int strike_x100        = (int)get_param(data, "strike_x100");
    int spot_x100          = (int)get_param(data, "spot_x100");
    int time_x1000         = (int)get_param(data, "time_x1000");
    ctx->N = (int)get_param(data, "N");
    ctx->num_steps = (int)get_param(data, "num_steps");
    ctx->risk_free = (float)risk_free_x10000 / 10000.0f;
    ctx->volatility = (float)volatility_x10000 / 10000.0f;
    ctx->strike = (float)strike_x100 / 100.0f;
    ctx->spot = (float)spot_x100 / 100.0f;
    ctx->time_to_maturity = (float)time_x1000 / 1000.0f;
    ctx->base_seed = (unsigned int)get_param(data, "base_seed");
    ctx->payoffs = (float*)calloc((size_t)(ctx->N), sizeof(float));
    return ctx;
}

void task_run(void* test_data) {
    MCContext* ctx = (MCContext*)test_data;
    solution_compute(ctx->N, ctx->num_steps, ctx->risk_free, ctx->volatility, ctx->strike, ctx->spot, ctx->time_to_maturity, ctx->base_seed, ctx->payoffs);
}

void task_write_output(void* test_data, const char* output_path) {
    MCContext* ctx = (MCContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->N; i++)
        fprintf(f, "%.6f\n", ctx->payoffs[i]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    MCContext* ctx = (MCContext*)test_data;
    free(ctx->payoffs);
    free(ctx);
}
