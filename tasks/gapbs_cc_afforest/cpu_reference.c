#include <stdint.h>
#include <stdlib.h>

static int g_n = 0;
static int g_neighbor_rounds = 0;
static int g_num_samples = 0;
static const int *g_row_ptr = NULL;
static const int *g_col_idx = NULL;
static int *g_comp = NULL;

static void LinkVertices(int u, int v, int *comp) {
    int p1 = comp[u];
    int p2 = comp[v];
    while (p1 != p2) {
        int high = (p1 > p2) ? p1 : p2;
        int low = p1 + (p2 - high);
        int p_high = comp[high];
        if (p_high == low) {
            break;
        }
        if (p_high == high) {
            comp[high] = low;
            break;
        }
        p1 = comp[comp[high]];
        p2 = comp[low];
    }
}

static void CompressAll(int n, int *comp) {
    for (int i = 0; i < n; ++i) {
        while (comp[i] != comp[comp[i]]) {
            comp[i] = comp[comp[i]];
        }
    }
}

static int SampleFrequentElement(const int *comp, int n, int num_samples) {
    if (n <= 0) return 0;
    if (num_samples <= 0) num_samples = 1;
    int table_cap = 1;
    while (table_cap < num_samples * 4) table_cap <<= 1;
    int *keys = (int*)malloc((size_t)table_cap * sizeof(int));
    int *counts = (int*)malloc((size_t)table_cap * sizeof(int));
    if (!keys || !counts) {
        free(keys);
        free(counts);
        return comp[0];
    }
    for (int i = 0; i < table_cap; ++i) {
        keys[i] = -1;
        counts[i] = 0;
    }

    uint32_t seed = 1u;
    for (int i = 0; i < num_samples; ++i) {
        seed = seed * 1664525u + 1013904223u;
        int idx = (int)(seed % (uint32_t)n);
        int key = comp[idx];
        int slot = (key * 2654435761u) & (uint32_t)(table_cap - 1);
        while (1) {
            if (keys[slot] == -1) {
                keys[slot] = key;
                counts[slot] = 1;
                break;
            }
            if (keys[slot] == key) {
                counts[slot] += 1;
                break;
            }
            slot = (slot + 1) & (table_cap - 1);
        }
    }

    int best_key = comp[0];
    int best_count = -1;
    for (int i = 0; i < table_cap; ++i) {
        if (keys[i] != -1 && counts[i] > best_count) {
            best_key = keys[i];
            best_count = counts[i];
        }
    }
    free(keys);
    free(counts);
    return best_key;
}

void solution_init(int n,
                   int neighbor_rounds,
                   int num_samples,
                   const int *row_ptr,
                   const int *col_idx) {
    g_n = n;
    g_neighbor_rounds = neighbor_rounds;
    g_num_samples = num_samples;
    g_row_ptr = row_ptr;
    g_col_idx = col_idx;
    if (g_comp) {
        free(g_comp);
        g_comp = NULL;
    }
    if (n > 0) {
        g_comp = (int*)malloc((size_t)n * sizeof(int));
    }
}

void solution_compute(int *comp_out) {
    if (!comp_out || g_n <= 0 || !g_row_ptr || !g_col_idx || !g_comp) {
        return;
    }

    for (int i = 0; i < g_n; ++i) {
        g_comp[i] = i;
    }

    for (int r = 0; r < g_neighbor_rounds; ++r) {
        for (int u = 0; u < g_n; ++u) {
            int e = g_row_ptr[u] + r;
            if (e < g_row_ptr[u + 1]) {
                LinkVertices(u, g_col_idx[e], g_comp);
            }
        }
        CompressAll(g_n, g_comp);
    }

    int skip_label = SampleFrequentElement(g_comp, g_n, g_num_samples);

    for (int u = 0; u < g_n; ++u) {
        if (g_comp[u] == skip_label) {
            continue;
        }
        int begin = g_row_ptr[u] + g_neighbor_rounds;
        int end = g_row_ptr[u + 1];
        if (begin > end) begin = end;
        for (int e = begin; e < end; ++e) {
            LinkVertices(u, g_col_idx[e], g_comp);
        }
    }

    CompressAll(g_n, g_comp);
    for (int i = 0; i < g_n; ++i) {
        comp_out[i] = g_comp[i];
    }
}

void solution_free(void) {
    g_n = 0;
    g_neighbor_rounds = 0;
    g_num_samples = 0;
    g_row_ptr = NULL;
    g_col_idx = NULL;
    if (g_comp) {
        free(g_comp);
        g_comp = NULL;
    }
}
