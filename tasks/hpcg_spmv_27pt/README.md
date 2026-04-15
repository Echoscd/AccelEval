# HPCG 27-Point Stencil Sparse Matrix-Vector Multiply

This task is derived from the official HPCG benchmark. HPCG builds a regular sparse linear system similar to a 3D heat-diffusion discretization, then uses that matrix inside a multigrid-preconditioned conjugate gradient benchmark. One central kernel is sparse matrix-vector multiplication (SpMV).

The official reference generator uses a 27-point 3D stencil: each grid point couples to all neighbors in a `3 x 3 x 3` neighborhood, with diagonal value `26.0` and off-diagonal values `-1.0`. The reference SpMV kernel computes `y = A x` over this sparse structure.

## Source
- `src/GenerateProblem_ref.cpp` for the 27-point matrix construction
- `src/ComputeSPMV_ref.cpp` for the reference SpMV kernel

## Why it fits GPU acceleration
SpMV is a classic sparse-linear-algebra kernel with row-level parallelism but strong memory-bandwidth pressure and irregular gathers from `x[col_idx[j]]`. This HPCG-derived instance is especially useful because its matrix has a regular 27-point origin while still being stored as CSR.

## Input
Parameters: `nx, ny, nz, n, nnz`

Tensors:
- `row_ptr` (`int32[n+1]`): CSR row offsets
- `col_idx` (`int32[nnz]`): CSR column indices
- `values` (`float64[nnz]`): nonzero values
- `x` (`float64[n]`): dense input vector

## Output
- `y` (`float64[n]`): output vector `A x`

## ORBench adaptation note
The official HPCG generator initializes the exact solution vector to all ones. This task follows that convention and uses `x = 1`.
