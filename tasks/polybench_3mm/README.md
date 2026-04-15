# PolyBench 3MM (Three Matrix Multiplications)

## Background

This task is adapted from the 3MM benchmark in the PolyBench/C suite.
It evaluates a chained dense linear algebra kernel made of three matrix
multiplications:

- `E = A * B`
- `F = C * D`
- `G = E * F`

Such multi-stage dense matrix products appear in scientific computing,
linear algebra workloads, and compiler/accelerator optimization studies.

## Source

PolyBench/C benchmark suite, 3MM benchmark
(`linear-algebra/kernels/3mm/3mm.c`).

## Why it fits GPU acceleration

3MM is dominated by dense matrix multiplication, which exposes abundant data
parallelism and regular memory access. Each stage can be implemented with
GPU GEMM-like kernels, shared-memory tiling, and register blocking. The
benchmark is also useful for evaluating whether multiple GEMM stages should
be executed separately or pipelined.

## Inputs

- `A`: float32 matrix of shape `[ni, nk]`, row-major.
- `B`: float32 matrix of shape `[nk, nj]`, row-major.
- `C`: float32 matrix of shape `[nj, nm]`, row-major.
- `D`: float32 matrix of shape `[nm, nl]`, row-major.
- Params:
  - `ni`, `nj`, `nk`, `nl`, `nm`: matrix dimensions.

## Outputs

- `G`: float32 matrix of shape `[ni, nl]`, row-major, equal to
  `(A * B) * (C * D)`.

The ORBench text output writes the final `G` matrix in row-major order, one
value per line.
