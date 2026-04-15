# PathFinder Grid Shortest-Path Dynamic Programming

## Problem Background

PathFinder solves a constrained shortest-path problem on a 2D weighted grid using dynamic programming.
Starting from any cell in the first row, the path moves one row downward at a time. At each step,
it may stay in the same column, move diagonally left, or move diagonally right. The goal is to compute,
for every ending column in the last row, the minimum accumulated path cost.

This is a classic wavefront dynamic-programming pattern:

```text
DP[r][c] = wall[r][c] + min(DP[r-1][c-1], DP[r-1][c], DP[r-1][c+1])
```

with boundary checks at the left and right edges.

## Algorithm Source

This task is adapted from the real GitHub benchmark source:
- Repository: `yuhc/gpu-rodinia`
- Benchmark: `openmp/pathfinder/pathfinder.cpp`
- Suite: Rodinia Benchmark Suite (University of Virginia)

The ORBench version keeps the same core recurrence and row-by-row update structure,
while converting the benchmark into ORBench's `input.bin` / `expected_output.txt` format.

## Why It Fits GPU Acceleration

- **Row-wise parallelism**: within each row, every column update is independent once the previous row is known.
- **Regular memory access**: the recurrence reads three neighboring values from the previous row and one grid value.
- **Double buffering**: only two row buffers are needed, which maps naturally to GPU global/shared memory.
- **Tiling opportunity**: neighboring-column dependencies are local, so shared-memory tiles with halo cells are natural.

The main bottleneck is synchronization across rows: row `r` cannot start until row `r-1` is finished.
That makes this task a good test of wavefront scheduling, buffering, and launch-overhead control.

## Input Format

Binary file `input.bin` (ORBench v2 format):

| Tensor | Type | Shape | Description |
|---|---|---:|---|
| `wall` | int32 | `[rows * cols]` | Row-major grid of nonnegative cell costs |

| Parameter | Type | Description |
|---|---|---|
| `rows` | int64 | Number of rows in the grid |
| `cols` | int64 | Number of columns in the grid |
| `seed` | int64 | RNG seed used by the data generator |

Cell costs are integers in `[0, 9]`.

## Output Format

`expected_output.txt` contains `cols` lines. Line `c` is the minimum path cost ending at column `c` of the last row.

```text
Format: "%d\n" per line
```

## Data Sizes

| Size | rows | cols | Cells |
|---|---:|---:|---:|
| small | 2048 | 512 | 1.05M |
| medium | 8192 | 1024 | 8.39M |
| large | 32768 | 2048 | 67.11M |

## Notes for ORBench Integration

- Recommended interface mode: `init_compute`
- Correctness check: exact integer match (`atol = 0`)
- Output length: `cols`
