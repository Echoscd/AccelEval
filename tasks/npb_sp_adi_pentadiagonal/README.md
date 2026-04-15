# NPB SP Structured ADI Pentadiagonal Sweep

This ORBench task is **adapted from the NAS Parallel Benchmarks (NPB) SP benchmark family** in the GitHub repository `benchmark-subsetting/NPB3.0-omp-C`, especially the SP benchmark pipeline organized around `compute_rhs`, `txinvr`, `x_solve`, `y_solve`, `z_solve`, and `add` in `SP/sp.c`.

The original NPB SP benchmark is one of NPB's three pseudo-applications and is classified by NASA as the **Scalar Penta-diagonal solver**. The OpenMP C repository's change log explicitly describes SP in terms of splitting `adi` into `compute_rhs`, `txinvr`, `x_solve and ninvr`, `y_solve and tzetar`, `z_solve and pinvr`, and `add`.

This ORBench task keeps that same algorithmic shape — repeated **residual construction + local transforms + directional scalar line solves** on a 3D structured grid with **5 coupled unknowns per point** — while repackaging it as a self-contained benchmark with generated binary inputs and a pure-C CPU reference.

## Problem background

We solve a synthetic structured-grid implicit update problem using an **ADI-style iteration**. Each grid point stores a 5-component state vector. A single iteration consists of:

- a `compute_rhs`-style residual build against a 3D pentadiagonal stencil operator,
- a `txinvr`-style local variable transform,
- an x-direction scalar pentadiagonal line solve followed by a `ninvr`-style transform,
- a y-direction scalar pentadiagonal line solve followed by a `tzetar`-style transform,
- a z-direction scalar pentadiagonal line solve followed by a `pinvr`-style transform,
- and a final `add`-style update of the solution.

Unlike the LU family, the directional solves here are **scalar line solves** rather than tiny block solves, which is what makes this task SP-flavored rather than LU-flavored.

## Why it is suitable for GPU acceleration

This benchmark exposes several GPU-relevant structures at once:

- many independent x/y/z line solves that can be batched,
- high-throughput structured-grid residual construction,
- repeated pointwise transforms between sweeps,
- and a final reduction for componentwise residual norms.

The main difficulty is that the directional sweeps are line-oriented and bandwidth sensitive, so performance depends on layout, batching, and line-buffer strategies rather than only on raw thread count.

## Input

- `u0`: float64 array of length `n*n*n*5`, initial state field in row-major `(k,j,i,m)` order
- `rhs`: float64 array of length `n*n*n*5`, target right-hand side / forcing field in the same order
- parameters:
  - `n`: structured grid size along each dimension
  - `iters`: number of ADI iterations
  - `omega_milli`: relaxation factor encoded as `omega * 1000`

## Output

- `residual_norms`: 5 float64 values, one per component, containing the final RMS residual after the last iteration

## Source provenance

Adapted from the NPB SP benchmark family in:
- GitHub repo: `benchmark-subsetting/NPB3.0-omp-C`
- benchmark directory: `SP/`
- key source file: `SP/sp.c`

This ORBench version is **not** a bit-for-bit port of the full NPB SP pseudo-application. Instead, it extracts the SP benchmark's **structured-grid ADI + scalar pentadiagonal solve flavor** and packages it into ORBench's `init_compute` format with explicit generated state and forcing inputs.
