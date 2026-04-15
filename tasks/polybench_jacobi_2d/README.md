# PolyBench Jacobi-2D (2-D Jacobi Stencil)

## Background

This task is adapted from the Jacobi-2D benchmark in the PolyBench/C suite.
It performs iterative stencil smoothing on a 2-D grid by repeatedly averaging
five-point neighborhoods. Jacobi-style stencil updates are a standard building
block in scientific computing, PDE solvers, and image/field smoothing.

## Source

PolyBench/C benchmark suite, Jacobi-2D benchmark
(`stencils/jacobi-2d/jacobi-2d.c`).

## Why it fits GPU acceleration

Jacobi-2D exposes regular, high-throughput parallelism over grid points. Each
interior point update depends only on nearby neighbors from the previous field,
so GPUs can map the stencil to thread-parallel kernels with coalesced memory
accesses and tiling/shared-memory reuse.

## Inputs

- `A0`: float32 matrix of shape `[n, n]`, row-major.
- `B0`: float32 matrix of shape `[n, n]`, row-major.
- Params:
  - `n`: grid size.
  - `tsteps`: number of PolyBench outer iterations.

## Outputs

- `A`: float32 matrix of shape `[n, n]`, row-major, equal to the final field
  after all Jacobi updates.

The ORBench text output writes the final `A` matrix in row-major order, one
value per line.
