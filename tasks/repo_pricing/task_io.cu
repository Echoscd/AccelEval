// task_io.cu — unified compute_only interface (auto-migrated)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

extern "C" void solution_compute(int N,
                             const int* settle_year,
                             const int* settle_month,
                             const int* settle_day,
                             const int* delivery_year,
                             const int* delivery_month,
                             const int* delivery_day,
                             const int* issue_year,
                             const int* issue_month,
                             const int* issue_day,
                             const int* maturity_year,
                             const int* maturity_month,
                             const int* maturity_day,
                             const float* bond_rates,
                             const float* repo_rates,
                             const float* bond_clean_prices,
                             const float* dummy_strikes,
                             float* prices);

typedef struct {
    int N;
    const int* settle_year;
    const int* settle_month;
    const int* settle_day;
    const int* delivery_year;
    const int* delivery_month;
    const int* delivery_day;
    const int* issue_year;
    const int* issue_month;
    const int* issue_day;
    const int* maturity_year;
    const int* maturity_month;
    const int* maturity_day;
    const float* bond_rates;
    const float* repo_rates;
    const float* bond_clean_prices;
    const float* dummy_strikes;
    float* prices;
} RepoContext;

extern "C" void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    RepoContext* ctx = (RepoContext*)calloc(1, sizeof(RepoContext));
    if (!ctx) return NULL;
    ctx->N = (int)get_param(data, "N");
    ctx->settle_year = get_tensor_int(data, "settle_year");
    ctx->settle_month = get_tensor_int(data, "settle_month");
    ctx->settle_day = get_tensor_int(data, "settle_day");
    ctx->delivery_year = get_tensor_int(data, "delivery_year");
    ctx->delivery_month = get_tensor_int(data, "delivery_month");
    ctx->delivery_day = get_tensor_int(data, "delivery_day");
    ctx->issue_year = get_tensor_int(data, "issue_year");
    ctx->issue_month = get_tensor_int(data, "issue_month");
    ctx->issue_day = get_tensor_int(data, "issue_day");
    ctx->maturity_year = get_tensor_int(data, "maturity_year");
    ctx->maturity_month = get_tensor_int(data, "maturity_month");
    ctx->maturity_day = get_tensor_int(data, "maturity_day");
    ctx->bond_rates = get_tensor_float(data, "bond_rates");
    ctx->repo_rates = get_tensor_float(data, "repo_rates");
    ctx->bond_clean_prices = get_tensor_float(data, "bond_clean_prices");
    ctx->dummy_strikes = get_tensor_float(data, "dummy_strikes");

    if (!ctx->settle_year || !ctx->settle_month || !ctx->settle_day || !ctx->delivery_year || !ctx->delivery_month || !ctx->delivery_day || !ctx->issue_year || !ctx->issue_month || !ctx->issue_day || !ctx->maturity_year || !ctx->maturity_month || !ctx->maturity_day || !ctx->bond_rates || !ctx->repo_rates || !ctx->bond_clean_prices || !ctx->dummy_strikes) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        free(ctx);
        return NULL;
    }
    ctx->prices = (float*)calloc((size_t)(ctx->N * 12), sizeof(float));
    return ctx;
}

extern "C" void task_run(void* test_data) {
    RepoContext* ctx = (RepoContext*)test_data;
    solution_compute(ctx->N, ctx->settle_year, ctx->settle_month, ctx->settle_day, ctx->delivery_year, ctx->delivery_month, ctx->delivery_day, ctx->issue_year, ctx->issue_month, ctx->issue_day, ctx->maturity_year, ctx->maturity_month, ctx->maturity_day, ctx->bond_rates, ctx->repo_rates, ctx->bond_clean_prices, ctx->dummy_strikes, ctx->prices);
}

extern "C" void task_write_output(void* test_data, const char* output_path) {
    RepoContext* ctx = (RepoContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->N; i++)
        fprintf(f, "%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                ctx->prices[i*12+0], ctx->prices[i*12+1],
                ctx->prices[i*12+2], ctx->prices[i*12+3],
                ctx->prices[i*12+4], ctx->prices[i*12+5],
                ctx->prices[i*12+6], ctx->prices[i*12+7],
                ctx->prices[i*12+8], ctx->prices[i*12+9],
                ctx->prices[i*12+10], ctx->prices[i*12+11]);
    fclose(f);
}

extern "C" void task_cleanup(void* test_data) {
    if (!test_data) return;
    RepoContext* ctx = (RepoContext*)test_data;
    free(ctx->prices);
    free(ctx);
}
