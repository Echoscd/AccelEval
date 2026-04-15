# GAPBS PageRank Pull Gauss-Seidel

## Problem background

This task computes PageRank scores on a directed graph. PageRank assigns each
vertex a score based on the scores of vertices pointing into it, with damping
that models random teleportation. The implementation here follows the GAP
Benchmark Suite pull-direction PageRank kernel.

## Algorithm source

Source family: GAP Benchmark Suite (GAPBS), `src/pr.cc`.
The GAPBS reference implementation uses an iterative pull-direction update and
lets newly updated scores become visible immediately, i.e. a Gauss-Seidel-like
variant.

## Why it fits GPU acceleration

The dominant cost is repeatedly traversing incoming edges and accumulating
neighbor contributions. This exposes parallelism across destination vertices and
across incoming edges inside high-degree vertices. GPU-friendly storage is an
incoming-edge CSR together with out-degree metadata.

## Input format

- `in_row_ptr` (`int32[n+1]`): CSR row offsets for incoming neighborhoods
- `in_col_idx` (`int32[m]`): source vertex IDs of incoming edges
- `out_degree` (`int32[n]`): out-degree of each vertex
- params:
  - `n`: number of vertices
  - `m`: number of directed edges
  - `max_iters`: maximum PageRank iterations

## Output format

- `scores` (`float32[n]`): final PageRank score of every vertex
