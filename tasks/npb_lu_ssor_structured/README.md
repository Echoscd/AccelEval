# NPB LU Structured SSOR Sweep

This ORBench task is **adapted from the NAS Parallel Benchmarks (NPB) LU benchmark** in the GitHub repository `benchmark-subsetting/NPB3.0-omp-C`, especially the LU benchmark's **SSOR / lower-triangular / upper-triangular solver structure** exposed through the `ssor`, `blts`, and `buts` routines in `LU/lu.c`.

The original NPB LU benchmark is a pseudo-application derived from CFD. In the NPB benchmark suite, LU is listed as a **Lower-Upper symmetric Gauss-Seidel solver**. The OpenMP C translation retains the LU benchmark directory and the main solver pipeline built around `ssor`, while the source file declares and uses the `blts` and `buts` triangular-solve kernels.

This ORBench task keeps that same algorithmic shape — repeated **lexicographic forward and backward block sweeps** on a 3D structured grid with **5 coupled unknowns per grid point** — but repackages it as a self-contained benchmark with generated binary inputs and a pure-C CPU reference.

## Problem background

We solve a synthetic structured-grid block system arising from a simplified CFD-style discretization. Each interior grid point stores a 5-component state vector. A single SSOR iteration consists of:

- a forward lexicographic lower/diagonal update,
- a backward lexicographic upper/diagonal update,
- and a final residual evaluation.

The local point update requires solving a small dense **5x5 block system** while coupling each variable to its six axis-aligned neighbors on the 3D grid.

## Why it is suitable for GPU acceleration

This benchmark combines several GPU-relevant ingredients:

- large structured-grid stencil-style neighbor access,
- repeated 3D sweeps over the full domain,
- tiny dense 5x5 block solves per grid point,
- substantial arithmetic intensity,
- and a final residual-norm reduction.

The main challenge is that the lexicographic SSOR order introduces directional dependencies, so parallelism is more limited than for fully explicit stencil updates. That makes it a useful benchmark for testing wavefront, coloring, or block-based GPU strategies.

## Input

- `u0`: float64 array of length `n*n*n*5`, initial state field in row-major `(k,j,i,m)` order
- `rhs`: float64 array of length `n*n*n*5`, right-hand-side / forcing field in the same order
- parameters:
  - `n`: interior grid size along each dimension
  - `iters`: number of SSOR iterations
  - `omega_milli`: relaxation factor encoded as `omega * 1000`

## Output

- `residual_norms`: 5 float64 values, one per coupled component, containing the final RMS residual after the last SSOR iteration

## Source provenance

Adapted from the NPB LU benchmark family in:
- GitHub repo: `benchmark-subsetting/NPB3.0-omp-C`
- benchmark directory: `LU/`
- key source file: `LU/lu.c`

This ORBench version is **not** a bit-for-bit port of the full NPB LU class-driven CFD solver. Instead, it extracts the LU benchmark's **structured-grid lower-upper SSOR flavor** and packages it into ORBench's `init_compute` format with explicit generated state and forcing inputs.
