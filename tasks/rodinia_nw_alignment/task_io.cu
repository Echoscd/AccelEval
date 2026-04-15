#include "orbench_io.h"
#include <cstdio>
#include <cstdlib>

extern "C" void solution_init(int seq_len, int penalty, const int* h_seq_a, const int* h_seq_b);
extern "C" void solution_compute(int* h_score_out);
extern "C" void solution_free(void);

typedef struct {
    int score_out[1];
} NWContext;

extern "C" void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    NWContext* ctx = (NWContext*)calloc(1, sizeof(NWContext));
    int seq_len = (int)get_param(data, "seq_len");
    int penalty = (int)get_param(data, "penalty");
    solution_init(seq_len, penalty, get_tensor_int(data, "seq_a"), get_tensor_int(data, "seq_b"));
    return ctx;
}

extern "C" void task_run(void* test_data) {
    NWContext* ctx = (NWContext*)test_data;
    solution_compute(ctx->score_out);
}

extern "C" void task_write_output(void* test_data, const char* output_path) {
    NWContext* ctx = (NWContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    fprintf(f, "%d\n", ctx->score_out[0]);
    fclose(f);
}

extern "C" void task_cleanup(void* test_data) {
    solution_free();
    free(test_data);
}
