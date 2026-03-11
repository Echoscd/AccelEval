// ORBench v2 smoke-test solution for bellman_ford
//
// Implements task-specific interface:
//   solution_setup(const TaskData* data, const char* data_dir)
//   solution_run(void* test_data)
//   solution_write_output(void* test_data, const char* output_path)
//   solution_cleanup(void* test_data)
//
// NOTE: This implementation is CPU-based (runs inside .cu) for correctness-first.
// It still exercises the GPU harness + framework pipeline end-to-end.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "orbench_io.h"

#ifndef INF_VAL
#define INF_VAL 1e30f
#endif

// Task-specific test data structure
typedef struct {
    // Graph data
    int V;
    int E;
    int* row_offsets;
    int* col_indices;
    float* weights;
    
    // Requests: (s, t) pairs
    int num_requests;
    int* sources;  // s values
    int* targets;  // t values
    
    // Output buffer
    float* distances;
    
    // Working buffer
    float* dist_buffer;  // size V, reused for each request
} BellmanFordTestData;

static void bellman_ford_cpu(int V, const int* row_offsets, const int* col_indices,
                             const float* weights, int source, float* dist) {
    for (int i = 0; i < V; i++) dist[i] = INF_VAL;
    if (source < 0 || source >= V) return;
    dist[source] = 0.0f;

    for (int round = 0; round < V - 1; round++) {
        int updated = 0;
        for (int u = 0; u < V; u++) {
            float du = dist[u];
            if (du >= INF_VAL) continue;
            int start = row_offsets[u];
            int end = row_offsets[u + 1];
            for (int idx = start; idx < end; idx++) {
                int v = col_indices[idx];
                float nd = du + weights[idx];
                if (nd < dist[v]) {
                    dist[v] = nd;
                    updated = 1;
                }
            }
        }
        if (!updated) break;
    }
}

extern "C" void* solution_setup(const TaskData* data, const char* data_dir) {
    BellmanFordTestData* td = (BellmanFordTestData*)calloc(1, sizeof(BellmanFordTestData));
    if (!td) {
        fprintf(stderr, "OOM allocating test data\n");
        return NULL;
    }
    
    // Load graph data from TaskData
    td->V = (int)get_param(data, "V");
    td->E = (int)get_param(data, "E");
    td->row_offsets = get_tensor_int(data, "row_offsets");
    td->col_indices = get_tensor_int(data, "col_indices");
    td->weights = get_tensor_float(data, "weights");
    
    if (td->V <= 0 || td->E <= 0 || !td->row_offsets || !td->col_indices || !td->weights) {
        fprintf(stderr, "Invalid TaskData\n");
        free(td);
        return NULL;
    }
    
    // Load requests.txt: task-specific format (s t pairs)
    char path[512];
    snprintf(path, sizeof(path), "%s/requests.txt", data_dir);
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Cannot open requests.txt\n");
        free(td);
        return NULL;
    }
    
    // Count requests
    char line[256];
    td->num_requests = 0;
    while (fgets(line, sizeof(line), f)) {
        int s, t;
        if (sscanf(line, "%d %d", &s, &t) == 2) {
            td->num_requests++;
        }
    }
    rewind(f);
    
    // Allocate request arrays
    td->sources = (int*)malloc((size_t)td->num_requests * sizeof(int));
    td->targets = (int*)malloc((size_t)td->num_requests * sizeof(int));
    td->distances = (float*)malloc((size_t)td->num_requests * sizeof(float));
    td->dist_buffer = (float*)malloc((size_t)td->V * sizeof(float));
    
    if (!td->sources || !td->targets || !td->distances || !td->dist_buffer) {
        fprintf(stderr, "OOM allocating request arrays\n");
        if (td->sources) free(td->sources);
        if (td->targets) free(td->targets);
        if (td->distances) free(td->distances);
        if (td->dist_buffer) free(td->dist_buffer);
        free(td);
        fclose(f);
        return NULL;
    }
    
    // Parse requests
    int idx = 0;
    while (fgets(line, sizeof(line), f)) {
        int s, t;
        if (sscanf(line, "%d %d", &s, &t) == 2) {
            td->sources[idx] = s;
            td->targets[idx] = t;
            idx++;
        }
    }
    fclose(f);
    
    return td;
}

extern "C" void solution_run(void* test_data) {
    BellmanFordTestData* td = (BellmanFordTestData*)test_data;
    
    // Process each (s, t) pair
    for (int r = 0; r < td->num_requests; r++) {
        int s = td->sources[r];
        int t = td->targets[r];
        
        // Compute shortest path from s to all nodes
        bellman_ford_cpu(td->V, td->row_offsets, td->col_indices, td->weights, s, td->dist_buffer);
        
        // Store distance from s to t
        td->distances[r] = (t >= 0 && t < td->V) ? td->dist_buffer[t] : INF_VAL;
    }
}

extern "C" void solution_write_output(void* test_data, const char* output_path) {
    BellmanFordTestData* td = (BellmanFordTestData*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) {
        fprintf(stderr, "Cannot open %s for writing\n", output_path);
        return;
    }
    
    // Write one distance per line (task-specific format)
    for (int i = 0; i < td->num_requests; i++) {
        fprintf(f, "%.6e\n", td->distances[i]);
    }
    
    fclose(f);
}

extern "C" void solution_cleanup(void* test_data) {
    if (!test_data) return;
    BellmanFordTestData* td = (BellmanFordTestData*)test_data;
    
    if (td->sources) free(td->sources);
    if (td->targets) free(td->targets);
    if (td->distances) free(td->distances);
    if (td->dist_buffer) free(td->dist_buffer);
    free(td);
}


