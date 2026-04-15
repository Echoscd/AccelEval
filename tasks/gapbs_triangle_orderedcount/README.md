# GAPBS Triangle Counting (OrderedCount)

## Problem background

This task counts the number of triangles in an undirected graph. A triangle is
an unordered 3-clique `(u, v, w)` where every pair of vertices shares an edge.
The task follows the ordered counting strategy used in the GAP Benchmark Suite.

## Algorithm source

Source family: GAP Benchmark Suite (GAPBS), triangle counting kernel
(`src/tc.cc`). The GAPBS reference implementation uses sorted neighborhoods and
counts each triangle exactly once through an ordering rule.

## Why it fits GPU acceleration

Triangle counting spends most of its time intersecting sorted neighborhoods.
That exposes parallelism across oriented edges `(u, v)` and across the merge
steps inside large neighborhood intersections. GPU implementations can exploit
parallel set intersections, load-balanced work assignment, and efficient
reductions of partial triangle counts.

## Input format

- `row_ptr` (`int32[n+1]`): CSR row offsets for the undirected graph.
- `col_idx` (`int32[m]`): sorted neighbor IDs for all vertices.
- params:
  - `n`: number of vertices.
  - `m`: number of directed CSR entries.

The generated graphs are undirected, duplicate-free, and have sorted
neighborhoods, matching the assumptions of the GAPBS kernel.

## Output format

- `triangle_count` (uint64 scalar): total number of triangles, counted exactly
  once using the orientation rule `u > v > w`.
