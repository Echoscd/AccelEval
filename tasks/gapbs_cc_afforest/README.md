# GAPBS Connected Components (Afforest)

This task is adapted from the GAP Benchmark Suite connected components kernel,
which uses the **Afforest** algorithm. Afforest first processes a sampled
subgraph, uses intermediate compression to approximate component structure, and
then skips the largest intermediate component during a final linking phase.

## Why it is suitable for GPU acceleration

- The sparse sampled rounds process one fixed neighbor offset per vertex.
- The final linking phase scans graph edges with abundant vertex- and edge-level
  parallelism.
- Path compression and component hooking introduce irregular memory access,
  making this a meaningful CPU→GPU graph analytics benchmark.

## Input

- `row_ptr` (`int32[n+1]`): CSR row offsets of an undirected graph
- `col_idx` (`int32[m]`): CSR adjacency entries
- params:
  - `n`: number of vertices
  - `m`: number of directed CSR edges
  - `neighbor_rounds`: number of sparse Afforest rounds
  - `num_samples`: deterministic sample count used to estimate the largest
    intermediate component

## Output

- `comp_out` (`int32[n]`): component label per vertex after the final Afforest
  compression step
