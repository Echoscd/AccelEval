# HPCG Symmetric Gauss-Seidel Sweep on a 27-Point Stencil Matrix

This task is derived from the official HPCG benchmark. HPCG builds a sparse linear system from a regular 3D 27-point stencil and uses a symmetric Gauss-Seidel (SYMGS) smoother inside its multigrid-preconditioned conjugate-gradient iteration.

The official reference SYMGS kernel performs one forward sweep followed by one backward sweep. It assumes each CSR row stores lower-triangular entries before the diagonal and upper-triangular entries after the diagonal.

## Source
- `src/GenerateProblem_ref.cpp` for the 27-point matrix construction
- `src/ComputeSYMGS_ref.cpp` for the reference symmetric Gauss-Seidel sweep

## Why it fits GPU acceleration
SYMGS is a classic sparse triangular-solve-like kernel with strong data dependencies, irregular memory access, and repeated reuse inside iterative solvers. It is a good stress test because naive parallelization is difficult, while level scheduling, coloring, or block-structured optimizations can substantially change performance.

## Input
Parameters: `nx, ny, nz, n, nnz`

Tensors:
- `row_ptr` (`int32[n+1]`): CSR row offsets
- `col_idx` (`int32[nnz]`): CSR column indices
- `values` (`float64[nnz]`): nonzero values
- `diag_idx` (`int32[n]`): CSR index of each diagonal element
- `rhs` (`float64[n]`): right-hand side vector
- `x_init` (`float64[n]`): initial guess vector

## Output
- `x` (`float64[n]`): result after one symmetric Gauss-Seidel sweep (forward then backward)

## ORBench adaptation note
This task uses the same 27-point matrix structure as the official HPCG reference generator and uses an all-ones exact vector to derive `rhs = A * 1`. The provided initial guess is the all-zero vector.
