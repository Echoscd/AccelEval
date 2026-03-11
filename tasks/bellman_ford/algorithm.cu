// algorithm.cu - Algorithm implementation for bellman_ford task
// This file contains ONLY the algorithm code (kernels and device functions)
// The I/O, timing, and main function are generated from templates.

// GPU implementation
void gpu_bellman_ford(const CSRGraph* g, int source, float* dist) {
    // >>> LLM CODE START <<<
    // LLM should implement the GPU Bellman-Ford algorithm here
    // >>> LLM CODE END <<<
}

// CPU reference implementation
void bellman_ford_cpu(const CSRGraph* g, int source, float* dist) {
    for (int i = 0; i < g->num_nodes; i++) dist[i] = INF_VAL;
    dist[source] = 0.0f;

    for (int round = 0; round < g->num_nodes - 1; round++) {
        int updated = 0;
        for (int u = 0; u < g->num_nodes; u++) {
            if (dist[u] >= INF_VAL) continue;
            for (int idx = g->row_offsets[u]; idx < g->row_offsets[u + 1]; idx++) {
                int v = g->col_indices[idx];
                float nd = dist[u] + g->weights[idx];
                if (nd < dist[v]) {
                    dist[v] = nd;
                    updated = 1;
                }
            }
        }
        if (!updated) break;
    }
}


