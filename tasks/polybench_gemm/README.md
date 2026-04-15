# PolyBench GEMM (C = alpha * A * B + beta * C)

## Background

This task is adapted from the GEMM benchmark in the PolyBench/C suite. The
kernel computes a dense matrix-matrix multiplication with accumulation:

C <- alpha * A * B + beta * C

It is one of the most standard building blocks in scientific computing,
linear algebra, simulation, optimization, and machine learning pipelines.

## Source

PolyBench/C benchmark suite, GEMM benchmark (`linear-algebra/blas/gemm/gemm.c`).

## Why it fits GPU acceleration

Dense GEMM exposes abundant parallelism across output tiles and inner-product
accumulations. The arithmetic intensity is high, and the kernel benefits from
shared-memory tiling, register blocking, coalesced loads, and fused multiply-add
throughput on GPUs.

## Inputs

- `A`: float32 matrix of shape `[ni, nk]`, row-major.
- `B`: float32 matrix of shape `[nk, nj]`, row-major.
- `C0`: float32 matrix of shape `[ni, nj]`, row-major. This is the initial value of `C`.
- Params:
  - `ni`, `nj`, `nk`: matrix dimensions.
  - `alpha_milli`, `beta_milli`: scalar coefficients scaled by 1000.

## Output

- `C`: float32 matrix of shape `[ni, nj]`, row-major, after applying
  `C = alpha * A * B + beta * C0`.
