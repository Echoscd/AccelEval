// harness_common.h - ORBench v2 request-based benchmark harness skeleton (C-only)
//
// Included by both framework/harness_gpu.cu and framework/harness_cpu.c
// so this file MUST be valid C (not C++).

#ifndef ORBENCH_HARNESS_COMMON_H
#define ORBENCH_HARNESS_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "orbench_io.h"

// Framework-agnostic: no task-specific assumptions

// Implemented by solution.cu / cpu_reference.c
// Framework-agnostic interface: solution defines its own test data structure
#ifdef __cplusplus
extern "C" {
#endif
// solution_setup: Load input.bin and requests.txt, allocate memory, return test_data
// test_data is task-specific and managed by solution
extern void* solution_setup(const TaskData* data, const char* data_dir);

// solution_run: Execute the computation (timed region)
extern void solution_run(void* test_data);

// solution_write_output: Write results to output.txt (task-specific format)
extern void solution_write_output(void* test_data, const char* output_path);

// solution_cleanup: Free all allocated resources
extern void solution_cleanup(void* test_data);
#ifdef __cplusplus
}
#endif

static int harness_main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <data_dir> [--validate]\n", argv[0]);
        return 1;
    }
    const char* data_dir = argv[1];
    int do_validate = (argc >= 3 && strcmp(argv[2], "--validate") == 0);

    // 1. Load input.bin
    char path[512];
    snprintf(path, sizeof(path), "%s/input.bin", data_dir);
    TaskData data = load_input_bin(path);

    // 2. Setup: solution loads requests.txt and allocates memory (not timed)
    void* test_data = solution_setup(&data, data_dir);
    if (!test_data) {
        fprintf(stderr, "solution_setup failed\n");
        free_task_data(&data);
        return 1;
    }

    // 3. Warmup (not timed)
    for (int w = 0; w < WARMUP; w++) {
        solution_run(test_data);
        SYNC();
    }

    // 4. Timed trials
    float total_ms = 0.0f, min_ms = 1e9f, max_ms = 0.0f;
    for (int t = 0; t < NUM_TRIALS; t++) {
        TIMER_START();
        solution_run(test_data);
        SYNC();
        TIMER_STOP();

        float ms = TIMER_ELAPSED_MS();
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
        if (ms > max_ms) max_ms = ms;
    }

    float mean_ms = total_ms / (float)NUM_TRIALS;
    printf("TIME_MS: %.3f\n", mean_ms);
    fprintf(stderr, "Timing: mean=%.3f ms, min=%.3f ms, max=%.3f ms (%d trials)\n",
            mean_ms, min_ms, max_ms, NUM_TRIALS);

    // 5. Validate: run once and write output.txt (task-specific format)
    if (do_validate) {
        solution_run(test_data);
        SYNC();
        snprintf(path, sizeof(path), "%s/output.txt", data_dir);
        solution_write_output(test_data, path);
    }

    // 6. Cleanup
    solution_cleanup(test_data);
    free_task_data(&data);
    return 0;
}

#endif // ORBENCH_HARNESS_COMMON_H


