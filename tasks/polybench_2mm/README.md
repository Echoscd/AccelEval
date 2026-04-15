# PolyBench 2MM (D = alpha * A * B * C + beta * D)

## Background

This task is adapted from the 2MM benchmark in the PolyBench/C suite. The
kernel performs two dense matrix multiplications in sequence:

1. `tmp <- alpha * A * B`
2. `D <- tmp * C + beta * D`

This pattern appears in blocked linear algebra pipelines and compound dense
numerical kernels where an intermediate product feeds another multiplication.

## Source

PolyBench/C benchmark suite, 2MM benchmark (`linear-algebra/kernels/2mm/2mm.c`).

## Why it fits GPU acceleration

2MM consists of two GEMM-like phases with high arithmetic intensity and regular
memory access. GPUs can exploit parallelism across output tiles in both phases,
using shared-memory tiling, register blocking, and efficient global-memory
access patterns. The intermediate matrix also creates opportunities for kernel
pipelining or carefully staged execution.

## Inputs

- `A`: float32 matrix of shape `[ni, nk]`, row-major.
- `B`: float32 matrix of shape `[nk, nj]`, row-major.
- `C`: float32 matrix of shape `[nj, nl]`, row-major.
- `D0`: float32 matrix of shape `[ni, nl]`, row-major. This is the initial value of `D`.
- Params:
  - `ni`, `nj`, `nk`, `nl`: matrix dimensions.
  - `alpha_milli`, `beta_milli`: scalar coefficients scaled by 1000.

## Output

- `D`: float32 matrix of shape `[ni, nl]`, row-major, after applying
  `D = (alpha * A * B) * C + beta * D0`.
