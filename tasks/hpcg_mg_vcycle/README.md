# HPCG Multigrid V-Cycle on a 27-Point Stencil Matrix

This task is derived from the official HPCG benchmark. HPCG builds a sparse linear
system from a regular 3D 27-point stencil and uses a multigrid preconditioner inside
its conjugate-gradient iteration. The multigrid reference implementation performs a
V-cycle with symmetric Gauss-Seidel smoothing, residual restriction to a coarse grid,
recursive coarse correction, and prolongation back to the fine grid.

## Source
- `src/GenerateProblem_ref.cpp` for the 27-point matrix construction
- `src/ComputeMG_ref.cpp` for the multigrid V-cycle structure
- `src/ComputeRestriction_ref.cpp` for the residual restriction rule
- `src/ComputeProlongation_ref.cpp` for the correction prolongation rule
- `src/ComputeSYMGS_ref.cpp` for the smoother used at each level

## Why it fits GPU acceleration
This task combines several important sparse iterative-solver kernels in one benchmark:
SpMV on a structured sparse matrix, Gauss-Seidel-like triangular sweeps, coarse-grid
residual formation, and nested-grid control flow. It stresses both memory bandwidth
and dependency management, and the regular 27-point structure also creates room for
specialized GPU kernels beyond generic CSR processing.

## Input
Parameters: `nx, ny, nz, n, nnz, coarse_levels`

Tensors:
- `row_ptr` (`int32[n+1]`): CSR row offsets of the finest matrix
- `col_idx` (`int32[nnz]`): CSR column indices of the finest matrix
- `values` (`float64[nnz]`): CSR nonzero values of the finest matrix
- `diag_idx` (`int32[n]`): CSR index of each diagonal entry
- `rhs` (`float64[n]`): right-hand side vector on the finest grid
- `x_init` (`float64[n]`): initial guess vector on the finest grid

## Output
- `x` (`float64[n]`): fine-grid solution after one multigrid V-cycle

## ORBench adaptation note
This task keeps the official HPCG 27-point structured matrix pattern on the finest
level and reconstructs the coarse levels geometrically during initialization. The
reference CPU implementation performs one presmoothing SYMGS sweep, injection-based
residual restriction, recursive coarse correction, prolongation back to mapped fine
points, and one postsmoothing SYMGS sweep per level.
