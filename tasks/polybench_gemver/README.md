# PolyBench GEMVER (Vector Multiplication and Matrix Addition)

## Background

This task is adapted from the PolyBench/C `gemver` benchmark. GEMVER performs a
rank-1-update style matrix modification followed by two dense matrix-vector
products. Starting from a dense square matrix `A`, it updates the matrix by
adding two outer products,

- `A := A + u1 * v1^T + u2 * v2^T`

then computes

- `x := beta * A^T * y + z`
- `w := alpha * A * x`

The kernel appears in dense linear algebra workloads and mixes matrix updates,
transpose-aware access, and matrix-vector multiplication.

## Source

Adapted from PolyBench/C `linear-algebra/blas/gemver/gemver.c`.

## Why it fits GPU acceleration

The outer-product matrix update exposes 2D elementwise parallelism over all
matrix entries. The two matrix-vector phases also offer row- or column-parallel
work with regular memory access patterns. The main bottlenecks are global memory
bandwidth and efficient reuse of matrix/vector data across the three phases.

## Input format

`input.bin` stores:

- `A`: float32 tensor of shape `[n, n]` in row-major order
- `u1`: float32 vector of length `n`
- `v1`: float32 vector of length `n`
- `u2`: float32 vector of length `n`
- `v2`: float32 vector of length `n`
- `y`: float32 vector of length `n`
- `z`: float32 vector of length `n`
- params: `n`

Constants follow PolyBench/C:

- `alpha = 1.5`
- `beta = 1.2`

## Output format

`output.txt` contains the output vector `w` of length `n`, one float per line.
