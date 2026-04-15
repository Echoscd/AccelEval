// cpu_reference.c — Held-Karp DP for TSP Tour Cost (CPU baseline)
//
// Faithfully ported from Google OR-Tools:
//   ortools/graph/hamiltonian_path.h
//     — HamiltonianPathSolver::Solve()             [lines 650–736]
//     — LatticeMemoryManager::Init()               [lines 362–394]
//     — LatticeMemoryManager::BaseOffset()          [lines 412–433]
//     — LatticeMemoryManager::OffsetDelta()         [lines 315–319]
//     — SetRangeWithCardinality / Gosper's hack     [lines 244–253]
//
// Reference: M. Held, R.M. Karp, "A dynamic programming approach to
// sequencing problems", J. SIAM 10 (1962) 196–210.
//
// Key features preserved from or-tools:
//   * LatticeMemoryManager with binomial-coefficient addressing for
//     cache-friendly sequential memory access (the main perf trick).
//   * Gosper's hack (MIT AI Memo 239, Item 175) to enumerate subsets
//     of a given cardinality in increasing order.
//   * Incremental OffsetDelta updates to avoid recomputing BaseOffset
//     for each (set, dest) pair — saves ~30-35% compute.
//
// NO file I/O, NO main(). All I/O handled by task_io_cpu.c.

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>

#define MAX_NODES 32  // matches Set<uint32_t>::kMaxCardinality

// ===== Bit utilities (matching ortools/util/bitset.h) =====

static int popcount32(uint32_t x) {
    // __builtin_popcount available in GCC/Clang
    return __builtin_popcount(x);
}

static int ctz32(uint32_t x) {
    return __builtin_ctz(x);
}

// ===== LatticeMemoryManager (lines 296–452) =====
// Stores f(set, node) with cache-friendly combinatorial addressing.
typedef struct {
    int max_card;
    uint64_t binomial[MAX_NODES + 1][MAX_NODES + 2];  // (n choose k)
    int64_t  base_offset[MAX_NODES + 1];
    int*     memory;
    size_t   memory_size;
} LatticeMemoryManager;

// Init() — lines 362–394
static void LatticeInit(LatticeMemoryManager* mem, int max_card) {
    int n, k;
    mem->max_card = max_card;

    // Pascal's triangle (lines 370–380)
    for (n = 0; n <= max_card; n++) {
        mem->binomial[n][0] = 1;
        for (k = 1; k <= n; k++) {
            mem->binomial[n][k] = mem->binomial[n - 1][k - 1]
                                + mem->binomial[n - 1][k];
        }
        mem->binomial[n][n + 1] = 0;  // extend for branchless code
    }

    // base_offset (lines 381–389)
    mem->base_offset[0] = 0;
    for (k = 0; k < max_card; k++) {
        mem->base_offset[k + 1] = mem->base_offset[k]
            + (int64_t)k * (int64_t)mem->binomial[max_card][k];
    }

    // Allocate memory (line 392)
    mem->memory_size = (size_t)max_card * (1ULL << (max_card - 1));
    mem->memory = (int*)malloc(mem->memory_size * sizeof(int));
}

// BaseOffset(card, set) — lines 412–433
// Computes offset using combinatorial number system.
static uint64_t LatticeBaseOffset(const LatticeMemoryManager* mem,
                                  int card, uint32_t set) {
    uint64_t local_offset = 0;
    int node_rank = 0;
    uint32_t s = set;
    while (s) {
        int node = ctz32(s);
        // binomial_coefficients_[node][node_rank + 1] (line 421)
        local_offset += mem->binomial[node][node_rank + 1];
        node_rank++;
        s &= s - 1;  // remove smallest element
    }
    return (uint64_t)mem->base_offset[card] + (uint64_t)card * local_offset;
}

// OffsetDelta(card, added_node, removed_node, rank) — lines 315–319
static int64_t LatticeOffsetDelta(const LatticeMemoryManager* mem,
                                  int card, int added_node,
                                  int removed_node, int rank) {
    return (int64_t)card *
           ((int64_t)mem->binomial[added_node][rank] -
            (int64_t)mem->binomial[removed_node][rank]);
}

// ElementRank(set, element) — SingletonRank
static int ElementRank(uint32_t set, int element) {
    // Count bits below 'element' in set
    return popcount32(set & ((1U << element) - 1));
}

// ===== Gosper's hack: next set with same popcount (lines 244–253) =====
// MIT AI Memo 239, Item 175 — "next higher number with same popcount"
static uint32_t GosperNext(uint32_t x) {
    uint32_t c = x & (uint32_t)(-(int32_t)x);  // smallest singleton
    uint32_t r = c + x;
    int shift = ctz32(x);
    return r == 0 ? 0 : (((r ^ x) >> (shift + 2)) | r);
}

// ===== HamiltonianPathSolver::Solve() — lines 650–736 =====
static int SolveTSP(int num_nodes, const int* cost_matrix) {
    LatticeMemoryManager mem;
    LatticeInit(&mem, num_nodes);

    // Initialize first layer (lines 665–668):
    // mem[{dest}, dest] = Cost(0, dest) for all dest
    for (int dest = 0; dest < num_nodes; dest++) {
        mem.memory[dest] = cost_matrix[0 * num_nodes + dest];
    }

    // DP iteration by cardinality (lines 672–706)
    for (int card = 2; card <= num_nodes; card++) {
        // Iterate over all subsets of size 'card' using Gosper's hack
        // (SetRangeWithCardinality, lines 268–283)
        uint32_t set_begin;
        uint32_t set_end;
        {
            // FullSet(card) = (1 << card) - 1
            set_begin = (1U << card) - 1;
            // end = FullSet(card-1).AddElement(num_nodes)
            // = ((1 << (card-1)) - 1) | (1 << num_nodes)
            set_end = ((1U << (card - 1)) - 1) | (1U << num_nodes);
        }

        for (uint32_t set = set_begin; set != set_end; set = GosperNext(set)) {
            // BaseOffset for this set (line 678)
            uint64_t set_offset = LatticeBaseOffset(&mem, card, set);

            // First subset = set.RemoveSmallestElement() (line 683)
            uint32_t first_subset = set & (set - 1);
            uint64_t subset_offset = LatticeBaseOffset(&mem, card - 1, first_subset);

            int prev_dest = ctz32(set);  // SmallestElement (line 684)
            int dest_rank = 0;

            // Iterate over destinations in set (line 686)
            uint32_t dest_iter = set;
            while (dest_iter) {
                int dest = ctz32(dest_iter);
                uint32_t subset = set & ~(1U << dest);  // set.RemoveElement(dest)

                int min_cost = INT_MAX;

                // Incremental offset update (line 692)
                subset_offset += LatticeOffsetDelta(&mem, card - 1,
                                                    prev_dest, dest, dest_rank);

                // Iterate over sources in subset (lines 694–699)
                int src_rank = 0;
                uint32_t src_iter = subset;
                while (src_iter) {
                    int src = ctz32(src_iter);
                    int cand = mem.memory[subset_offset + src_rank]
                             + cost_matrix[src * num_nodes + dest];
                    if (cand < min_cost) min_cost = cand;
                    src_rank++;
                    src_iter &= src_iter - 1;
                }

                prev_dest = dest;
                mem.memory[set_offset + dest_rank] = min_cost;
                dest_rank++;

                dest_iter &= dest_iter - 1;  // advance to next dest
            }
        }
    }

    // TSP cost = mem[full_set, 0] (line 712)
    uint32_t full_set = (1U << num_nodes) - 1;
    uint64_t full_offset = LatticeBaseOffset(&mem, num_nodes, full_set);
    int tsp_cost = mem.memory[full_offset + ElementRank(full_set, 0)];

    free(mem.memory);
    return tsp_cost;
}

// ===== Public interface =====

void solution_compute(int B, int n, const int* costs, int* tour_costs_out)
{
    for (int b = 0; b < B; b++) {
        const int* cost = costs + (size_t)b * (size_t)n * (size_t)n;
        tour_costs_out[b] = SolveTSP(n, cost);
    }
}

void solution_free(void) {}
