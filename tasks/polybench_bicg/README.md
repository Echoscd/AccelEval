# PolyBench BiCG (BiCG Sub Kernel)

## Background

This task is adapted from the BiCG benchmark in the PolyBench/C suite. The
kernel computes two coupled dense linear-algebra sub-operations over the same
matrix:

- `q <- A * p`
- `s <- A^T * r`

These operations appear inside Krylov iterative solvers such as BiCGStab,
where matrix-vector and transpose-matrix-vector products are fundamental inner
kernels.

## Source

PolyBench/C benchmark suite, BiCG benchmark
(`linear-algebra/kernels/bicg/bicg.c`).

## Why it fits GPU acceleration

BiCG exposes abundant data parallelism across matrix rows and columns. The
matrix-vector product and transpose-matrix-vector product both map naturally to
thread-parallel execution, and the kernel mixes regular memory accesses with
reduction-like accumulation. GPUs can accelerate both output vectors by
parallelizing over rows/columns and staging vector data efficiently.

## Inputs

- `A`: float32 matrix of shape `[n, m]`, row-major.
- `p`: float32 vector of length `m`.
- `r`: float32 vector of length `n`.
- Params:
  - `n`, `m`: matrix dimensions.

## Outputs

- `s`: float32 vector of length `m`, equal to `A^T * r`.
- `q`: float32 vector of length `n`, equal to `A * p`.

The ORBench text output writes all entries of `s` first, followed by all
entries of `q`.
