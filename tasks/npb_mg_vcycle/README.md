# NPB MG Multigrid V-Cycle (Structured 3D Poisson)

This ORBench task is **adapted from the NAS Parallel Benchmarks (NPB) MG benchmark** in the GitHub repository `benchmark-subsetting/NPB3.0-omp-C`, especially the MG core routine structure around `mg3P`, `resid`, `psinv`, `rprj3`, and `interp`.

The original MG benchmark is a geometric multigrid kernel on a structured 3D grid. In the source, `mg3P` is explicitly described as the **multigrid V-cycle routine**, while `psinv` is the smoother / approximate inverse and `resid` computes the residual. This ORBench task keeps that same algorithmic shape, but packages it as a self-contained benchmark with binary input generation and a pure-C CPU reference.

## Problem background

We solve a structured-grid 3D Poisson-like linear system using repeated multigrid V-cycles under zero Dirichlet boundary conditions. Multigrid is a classical algorithm in scientific computing because local smoothers quickly damp high-frequency error, while coarse-grid correction removes the remaining low-frequency error.

## Why it is suitable for GPU acceleration

The benchmark combines several GPU-friendly patterns:

- 3D stencil sweeps over a regular grid (`resid`, smoothing)
- restriction from fine to coarse grids
- prolongation / correction from coarse to fine grids
- repeated V-cycle application over multiple levels

Most of the work is bulk array processing with abundant parallelism over grid points. The main bottleneck is memory bandwidth and stencil data movement.

## Input

- `rhs`: float64 array of length `(n+2)^3`, stored in row-major order with halo / boundary cells included
- parameters:
  - `interior_n`: interior grid width/height/depth
  - `num_cycles`: number of V-cycles to run
  - `pre_smooth`: weighted-Jacobi sweeps before restriction
  - `post_smooth`: weighted-Jacobi sweeps after prolongation
  - `coarse_iters`: smoother iterations on the coarsest grid

## Output

- one float64 scalar: final L2 norm of the residual after all V-cycles

## Source provenance

Adapted from the NPB MG benchmark in:
- GitHub repo: `benchmark-subsetting/NPB3.0-omp-C`
- file: `MG/mg.c`

This ORBench version is **not** a bit-for-bit port of the full NPB class-driven driver. Instead, it extracts the benchmark's core multigrid structure and repackages it into ORBench's `init_compute` task format.
