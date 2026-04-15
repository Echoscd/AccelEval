# NPB CG Sparse Conjugate Gradient Solve

This ORBench task is **adapted from the NAS Parallel Benchmarks (NPB) CG benchmark** in the GitHub repository `benchmark-subsetting/NPB3.0-omp-C`, especially the CG benchmark's matrix-construction (`makea`) and conjugate-gradient (`conj_grad`) structure.

The original NPB CG benchmark is a sparse iterative linear-algebra kernel that stresses **irregular memory access and communication**. In the NPB suite, CG is listed as the conjugate-gradient kernel, and the OpenMP C translation explicitly highlights updates around `makea`, `conj_grad`, and explicit residual-norm computation.

This ORBench task keeps that same algorithmic shape — repeated sparse matrix-vector multiplies, dot products, and vector updates on a sparse symmetric positive definite system — but repackages it as a self-contained benchmark with generated CSR input, binary data files, and a pure-C CPU reference.

## Problem background

We solve a sparse symmetric positive definite linear system

\[
Ax = b
\]

using the classical conjugate-gradient algorithm on a matrix stored in CSR format.
This is one of the standard kernels in scientific computing and large-scale simulation, especially when sparse linear systems arise from discretized PDEs, graph problems, or optimization models.

## Why it is suitable for GPU acceleration

This benchmark combines several GPU-friendly ingredients:

- CSR sparse matrix-vector multiplication (SpMV)
- large parallel dot products / reductions
- SAXPY-style vector updates (`x += alpha p`, `r -= alpha Ap`)
- repeated iteration with fixed data structures

The main bottlenecks are memory bandwidth, irregular sparse access, and reduction efficiency.

## Input

- `row_ptr`: int32 array of length `n+1`
- `col_idx`: int32 array of length `nnz`
- `values`: float64 array of length `nnz`
- `b`: float64 array of length `n`
- parameters:
  - `n`: number of rows / columns
  - `nnz`: number of nonzeros
  - `max_iters`: maximum CG iterations
  - `tol_exp`: stopping tolerance encoded as `10^{-tol_exp}` on residual L2 norm

## Output

- `x`: float64 array of length `n`, the final approximate solution vector

## Source provenance

Adapted from the NPB CG benchmark in:
- GitHub repo: `benchmark-subsetting/NPB3.0-omp-C`
- benchmark directory: `CG/`

This ORBench version is **not** a bit-for-bit port of the full NPB class-driven inverse-iteration driver. Instead, it extracts the sparse-CG core and packages it into ORBench's `init_compute` format with explicit CSR input.
