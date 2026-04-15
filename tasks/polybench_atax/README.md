# PolyBench ATAX (A^T(Ax))

## Background

This task is adapted from the ATAX benchmark in the PolyBench/C suite.
ATAX computes a dense matrix-vector product followed by a transpose-matrix-
vector product:

- `tmp = A * x`
- `y = A^T * tmp`

This pattern appears in linear algebra pipelines and iterative numerical
methods where a matrix and its transpose are both applied.

## Source

PolyBench/C benchmark suite, ATAX benchmark
(`linear-algebra/kernels/atax/atax.c`).

## Why it fits GPU acceleration

ATAX has regular dense linear algebra access patterns and exposes abundant
parallelism across matrix rows and columns. GPUs can accelerate both stages
using matrix-vector kernels, reduction-friendly thread mappings, and memory
reuse strategies for `x` and `tmp`.

## Inputs

- `A`: float32 matrix of shape `[m, n]`, row-major.
- `x`: float32 vector of length `n`.
- Params:
  - `m`, `n`: matrix dimensions.

## Outputs

- `y`: float32 vector of length `n`, equal to `A^T * (A * x)`.

The ORBench text output writes the final `y` vector, one value per line.
