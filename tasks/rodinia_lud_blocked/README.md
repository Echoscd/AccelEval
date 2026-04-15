# Rodinia Blocked LU Decomposition (No Pivoting)

## Background
This task performs dense LU decomposition without pivoting on a square float32 matrix, using the blocked structure from the Rodinia LUD benchmark. The output stores the combined factors in-place: the strict lower triangle contains multipliers for `L` (with an implicit unit diagonal), while the upper triangle contains `U`.

## Source
Adapted from the Rodinia OpenMP LUD benchmark (`openmp/lud/omp/lud_omp.c`). The original implementation uses a fixed block size of 16 and updates diagonal, perimeter, and interior blocks in phases.

## Why it fits GPU acceleration
The diagonal block factorization is small and sequential within each 16x16 tile, but the perimeter block updates and especially the interior block updates expose large amounts of data parallelism. The interior phase resembles tiled matrix multiplication/subtraction, making it natural for GPU shared-memory tiling and warp-cooperative execution.

## Input
- `n` (param): matrix dimension, always a multiple of 16.
- `A0` (`float32`, length `n*n`): dense row-major input matrix.

## Output
- `LU` (`float32`, length `n*n`): dense row-major matrix containing the in-place LU factors.

## Notes
- The decomposition is **without pivoting**, so generated matrices are chosen to be numerically well-behaved.
- The ORBench data generator creates reproducible diagonally strengthened dense matrices.
