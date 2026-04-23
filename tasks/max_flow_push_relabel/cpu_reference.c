// cpu_reference.c — Push-Relabel Maximum Flow (CPU baseline)
//
// Faithfully ported from Google OR-Tools:
//   ortools/graph/generic_max_flow.h
//     — GenericMaxFlow::Solve()                     [lines 766–792]
//     — GenericMaxFlow::InitializePreflow()          [lines 794–837]
//     — GenericMaxFlow::Discharge()                  [lines 1270–1306]
//     — GenericMaxFlow::PushFlow()                   [lines 1139–1161]
//     — GenericMaxFlow::Relabel()                    [lines 1308–1337]
//     — GenericMaxFlow::GlobalUpdate()               [lines 1000–1095]
//     — GenericMaxFlow::RefineWithGlobalUpdate()      [lines 1202–1267]
//     — GenericMaxFlow::SaturateOutgoingArcsFromSource() [lines 1097–1137]
//
// Reference: A.V. Goldberg and R.E. Tarjan, "A new approach to the maximum
// flow problem", ACM STOC 1986.
//
// Graph representation: forward + reverse arcs with explicit opposite[] mapping.
// For E user arcs, total 2E arcs. Arc i's reverse is opposite[i].
// Adjacency stored as CSR: adj_start[node] .. adj_start[node+1].
//
// NO file I/O, NO main(). All I/O handled by task_io_cpu.c.

#include <stdlib.h>
#include <string.h>
#include <limits.h>

// ===== Module-level state =====
static int  g_num_nodes;
static int  g_num_total_arcs;  // 2 * E
static int  g_source;
static int  g_sink;

// Graph structure (built in solution_init from user arcs)
static int* g_arc_head;       // [2E]
static int* g_arc_opposite;   // [2E]
static int* g_adj_start;      // [num_nodes + 1]
static int* g_adj_list;       // [2E], arc indices sorted by tail

// Algorithm state (reset each solve)
static int*  g_residual;      // [2E] residual capacity
static int*  g_initial_cap;   // [2E] initial capacity (for reset)
static long long* g_excess;   // [num_nodes]
static int*  g_height;        // [num_nodes]
static int*  g_first_arc;     // [num_nodes] first admissible arc index in adj

// BFS queue for GlobalUpdate
static int*  g_bfs_queue;
static int*  g_in_queue;      // boolean

// Active node stack (simple LIFO for highest-label heuristic approximation)
static int*  g_active_stack;
static int   g_active_top;

// ===== PushFlow (lines 1139–1161) =====
static void PushFlow(int flow, int tail, int arc) {
    g_residual[arc] -= flow;
    g_residual[g_arc_opposite[arc]] += flow;
    g_excess[tail] -= flow;
    g_excess[g_arc_head[arc]] += flow;
}

// ===== Relabel (lines 1308–1337) =====
static void Relabel(int node) {
    int min_height = INT_MAX;
    int best_arc = -1;
    for (int i = g_adj_start[node]; i < g_adj_start[node + 1]; i++) {
        int arc = g_adj_list[i];
        if (g_residual[arc] > 0) {
            int h = g_height[g_arc_head[arc]];
            if (h < min_height) {
                min_height = h;
                best_arc = i;
                if (min_height + 1 == g_height[node]) break;
            }
        }
    }
    if (best_arc >= 0) {
        g_height[node] = min_height + 1;
        g_first_arc[node] = best_arc;
    }
}

// ===== Discharge (lines 1270–1306) =====
static void Discharge(int node) {
    int num_nodes = g_num_nodes;
    while (1) {
        for (int i = g_first_arc[node]; i < g_adj_start[node + 1]; i++) {
            int arc = g_adj_list[i];
            // IsAdmissible check (line 360)
            if (g_residual[arc] > 0 &&
                g_height[node] == g_height[g_arc_head[arc]] + 1) {
                int head = g_arc_head[arc];
                if (g_excess[head] == 0 && head != g_source && head != g_sink) {
                    // Push head as active
                    g_active_stack[g_active_top++] = head;
                }
                // PushAsMuchFlowAsPossible (lines 1163–1182)
                int flow = g_residual[arc];
                if ((long long)flow > g_excess[node]) flow = (int)g_excess[node];
                PushFlow(flow, node, arc);
                if (g_excess[node] == 0) {
                    g_first_arc[node] = i;
                    return;
                }
            }
        }
        Relabel(node);
        if (g_height[node] >= num_nodes) break;
    }
}

// ===== GlobalUpdate — BFS from sink in reverse residual (lines 1000–1095) =====
static void GlobalUpdate(void) {
    int num_nodes = g_num_nodes;
    memset(g_in_queue, 0, num_nodes * sizeof(int));
    g_in_queue[g_sink] = 1;
    g_in_queue[g_source] = 1;
    int qfront = 0, qback = 0;
    g_bfs_queue[qback++] = g_sink;

    while (qfront < qback) {
        int node = g_bfs_queue[qfront++];
        int candidate_dist = g_height[node] + 1;
        for (int i = g_adj_start[node]; i < g_adj_start[node + 1]; i++) {
            int arc = g_adj_list[i];
            int head = g_arc_head[arc];
            if (g_in_queue[head]) continue;
            int opp = g_arc_opposite[arc];
            if (g_residual[opp] > 0) {
                g_height[head] = candidate_dist;
                g_in_queue[head] = 1;
                g_bfs_queue[qback++] = head;
            }
        }
    }
    // Unreachable nodes get height 2n-1 (line 1081)
    for (int node = 0; node < num_nodes; node++) {
        if (!g_in_queue[node]) g_height[node] = 2 * num_nodes - 1;
    }
    // Reset active node stack
    g_active_top = 0;
    for (int i = 1; i < qback; i++) {
        int node = g_bfs_queue[i];
        if (node != g_source && node != g_sink && g_excess[node] > 0) {
            g_active_stack[g_active_top++] = node;
        }
    }
}

// ===== SaturateOutgoingArcsFromSource (lines 1097–1137) =====
static int SaturateSource(void) {
    int pushed = 0;
    for (int i = g_adj_start[g_source]; i < g_adj_start[g_source + 1]; i++) {
        int arc = g_adj_list[i];
        int flow = g_residual[arc];
        if (flow > 0 && g_height[g_arc_head[arc]] < g_num_nodes) {
            PushFlow(flow, g_source, arc);
            pushed = 1;
        }
    }
    return pushed;
}

// ===== InitializePreflow (lines 794–837) =====
static void InitializePreflow(void) {
    int num_nodes = g_num_nodes;
    int total_arcs = g_num_total_arcs;
    memset(g_excess, 0, num_nodes * sizeof(long long));
    memcpy(g_residual, g_initial_cap, total_arcs * sizeof(int));
    memset(g_height, 0, num_nodes * sizeof(int));
    g_height[g_source] = num_nodes;
    for (int node = 0; node < num_nodes; node++) {
        g_first_arc[node] = g_adj_start[node];
    }
    g_active_top = 0;
}

// ===== PushFlowExcessBackToSource (simplified) =====
// Push remaining excess back to source via reverse arcs.
static void PushFlowExcessBackToSource(void) {
    // Simple approach: for each node with excess, push back along reverse arcs
    for (int node = 0; node < g_num_nodes; node++) {
        if (node == g_source || node == g_sink) continue;
        while (g_excess[node] > 0) {
            int found = 0;
            for (int i = g_adj_start[node]; i < g_adj_start[node + 1]; i++) {
                int arc = g_adj_list[i];
                int opp = g_arc_opposite[arc];
                // Check if opposite arc has flow (residual > initial)
                if (g_residual[opp] < g_initial_cap[opp]) {
                    int flow_on_opp = g_initial_cap[opp] - g_residual[opp];
                    // This means there's flow on arc that we can cancel
                    int cancel = (int)(g_excess[node] < (long long)g_residual[arc]
                                       ? g_excess[node] : g_residual[arc]);
                    if (cancel <= 0) continue;
                    // We want to push flow on arc (from node towards head)
                    // But we need to push BACK, so we push on the opposite direction
                    // Check if there's flow from head→node (= residual[opp] < initial_cap[opp])
                    int back_flow = g_initial_cap[opp] - g_residual[opp];
                    if (back_flow <= 0) continue;
                    if (cancel > back_flow) cancel = back_flow;
                    PushFlow(cancel, node, arc);
                    found = 1;
                    if (g_excess[node] == 0) break;
                }
            }
            if (!found) break;
        }
    }
}

// ===== RefineWithGlobalUpdate (lines 1202–1267) =====
// Simplified: single round of saturate + discharge. For graphs without
// integer overflow (our benchmark), one round suffices to find the max flow
// value. PushFlowExcessBackToSource is only needed for flow decomposition
// or when sum of capacities exceeds int64 range.
static void RefineWithGlobalUpdate(void) {
    SaturateSource();
    GlobalUpdate();
    while (g_active_top > 0) {
        int node = g_active_stack[--g_active_top];
        if (g_excess[node] <= 0) continue;
        if (node == g_source || node == g_sink) continue;
        if (g_height[node] >= g_num_nodes) continue;
        Discharge(node);
    }
}

// ===== Solve (lines 766–792) =====
static long long Solve(void) {
    InitializePreflow();
    RefineWithGlobalUpdate();
    return g_excess[g_sink];
}

// ===== Public interface =====

static void _orbench_old_init(int  num_nodes,
                   int  num_arcs,       // E user arcs
                   const int* tails,    // [E]
                   const int* heads,    // [E]
                   const int* caps,     // [E]
                   int  source,
                   int  sink)
{
    g_num_nodes = num_nodes;
    g_num_total_arcs = 2 * num_arcs;
    g_source = source;
    g_sink = sink;

    int total = 2 * num_arcs;

    // Build arc arrays: first E are forward, next E are reverse
    g_arc_head     = (int*)malloc(total * sizeof(int));
    g_arc_opposite = (int*)malloc(total * sizeof(int));
    g_initial_cap  = (int*)malloc(total * sizeof(int));
    g_residual     = (int*)malloc(total * sizeof(int));

    for (int i = 0; i < num_arcs; i++) {
        g_arc_head[i]            = heads[i];            // forward
        g_arc_head[num_arcs + i] = tails[i];            // reverse
        g_arc_opposite[i]            = num_arcs + i;
        g_arc_opposite[num_arcs + i] = i;
        g_initial_cap[i]            = caps[i];           // forward has capacity
        g_initial_cap[num_arcs + i] = 0;                 // reverse has 0
    }

    // Build CSR adjacency: for each node, list all incident arcs (both forward and reverse)
    // Count degrees
    int* degree = (int*)calloc(num_nodes, sizeof(int));
    for (int a = 0; a < total; a++) {
        // tail of arc a:
        int tail;
        if (a < num_arcs) tail = tails[a];
        else              tail = heads[a - num_arcs];
        degree[tail]++;
    }
    g_adj_start = (int*)malloc((num_nodes + 1) * sizeof(int));
    g_adj_start[0] = 0;
    for (int i = 0; i < num_nodes; i++) {
        g_adj_start[i + 1] = g_adj_start[i] + degree[i];
    }
    g_adj_list = (int*)malloc(total * sizeof(int));
    memset(degree, 0, num_nodes * sizeof(int));
    for (int a = 0; a < total; a++) {
        int tail;
        if (a < num_arcs) tail = tails[a];
        else              tail = heads[a - num_arcs];
        g_adj_list[g_adj_start[tail] + degree[tail]] = a;
        degree[tail]++;
    }
    free(degree);

    // Allocate algorithm state
    g_excess       = (long long*)calloc(num_nodes, sizeof(long long));
    g_height       = (int*)calloc(num_nodes, sizeof(int));
    g_first_arc    = (int*)malloc(num_nodes * sizeof(int));
    g_bfs_queue    = (int*)malloc(num_nodes * sizeof(int));
    g_in_queue     = (int*)malloc(num_nodes * sizeof(int));
    g_active_stack = (int*)malloc(num_nodes * sizeof(int));
    g_active_top   = 0;
}

static void _orbench_old_compute(int num_nodes, int* max_flow_out)
{
    *max_flow_out = (int)Solve();
}

// ── Unified compute_only wrapper (auto-migrated) ──
void solution_compute(int num_nodes, int num_arcs, const int* tails, const int* heads, const int* caps, int source, int sink, int* max_flow_out) {
    _orbench_old_init(num_nodes, num_arcs, tails, heads, caps, source, sink);
    _orbench_old_compute(num_nodes, max_flow_out);
}
