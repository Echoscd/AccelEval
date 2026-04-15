#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

static int g_num_nodes = 0;
static int g_num_edges = 0;
static int g_source = 0;
static const int* g_node_start = NULL;
static const int* g_node_degree = NULL;
static const int* g_edge_dst = NULL;

static unsigned char* g_graph_mask = NULL;
static unsigned char* g_updating_graph_mask = NULL;
static unsigned char* g_graph_visited = NULL;
static int* g_cost = NULL;

void solution_init(int num_nodes, int num_edges, int source,
                   const int* node_start,
                   const int* node_degree,
                   const int* edge_dst) {
    g_num_nodes = num_nodes;
    g_num_edges = num_edges;
    g_source = source;
    g_node_start = node_start;
    g_node_degree = node_degree;
    g_edge_dst = edge_dst;

    g_graph_mask = (unsigned char*)calloc((size_t)num_nodes, sizeof(unsigned char));
    g_updating_graph_mask = (unsigned char*)calloc((size_t)num_nodes, sizeof(unsigned char));
    g_graph_visited = (unsigned char*)calloc((size_t)num_nodes, sizeof(unsigned char));
    g_cost = (int*)malloc((size_t)num_nodes * sizeof(int));

    for (int i = 0; i < num_nodes; ++i) g_cost[i] = -1;
    if (source >= 0 && source < num_nodes) {
        g_graph_mask[source] = 1;
        g_graph_visited[source] = 1;
        g_cost[source] = 0;
    }
}

void solution_compute(int* out_dist) {
    if (!g_graph_mask || !g_updating_graph_mask || !g_graph_visited || !g_cost) return;

    int stop;
    do {
        stop = 0;
        for (int tid = 0; tid < g_num_nodes; ++tid) {
            if (g_graph_mask[tid]) {
                g_graph_mask[tid] = 0;
                int start = g_node_start[tid];
                int deg = g_node_degree[tid];
                for (int i = start; i < start + deg; ++i) {
                    int id = g_edge_dst[i];
                    if (!g_graph_visited[id]) {
                        g_cost[id] = g_cost[tid] + 1;
                        g_updating_graph_mask[id] = 1;
                    }
                }
            }
        }
        for (int tid = 0; tid < g_num_nodes; ++tid) {
            if (g_updating_graph_mask[tid]) {
                g_graph_mask[tid] = 1;
                g_graph_visited[tid] = 1;
                stop = 1;
                g_updating_graph_mask[tid] = 0;
            }
        }
    } while (stop);

    memcpy(out_dist, g_cost, (size_t)g_num_nodes * sizeof(int));
}

void solution_free(void) {
    free(g_graph_mask);
    free(g_updating_graph_mask);
    free(g_graph_visited);
    free(g_cost);
    g_graph_mask = NULL;
    g_updating_graph_mask = NULL;
    g_graph_visited = NULL;
    g_cost = NULL;
    g_node_start = NULL;
    g_node_degree = NULL;
    g_edge_dst = NULL;
    g_num_nodes = 0;
    g_num_edges = 0;
    g_source = 0;
}

#ifdef __cplusplus
}
#endif
