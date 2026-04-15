# Rodinia Needleman-Wunsch Global Alignment

This task is adapted from the Rodinia benchmark suite's Needleman-Wunsch benchmark (`openmp/nw/needle.cpp`). It computes the optimal global alignment score between two protein-like sequences using dynamic programming with the BLOSUM62 substitution matrix and a constant gap penalty.

## Why it fits ORBench

Needleman-Wunsch is a classic dynamic-programming algorithm from bioinformatics. The dynamic-programming table has strong anti-diagonal / wavefront parallelism, which makes it a natural CPU→GPU benchmark: cells on the same anti-diagonal can be updated in parallel once their dependencies are ready.

Rodinia's implementation uses blocked 16×16 wavefront processing. This ORBench task keeps the same benchmark flavor:
- sequence lengths are multiples of 16,
- scoring uses BLOSUM62,
- gap penalty is fixed per instance,
- the workload is dominated by filling the DP table.

## Input

`input.bin` stores:
- `seq_a[int32, seq_len]`: first encoded sequence
- `seq_b[int32, seq_len]`: second encoded sequence

Parameters:
- `seq_len`: length of each sequence (multiple of 16)
- `penalty`: positive gap penalty

Each sequence element is an integer in `[0, 23]`, indexing the BLOSUM62 table.

## Output

A single integer:
- the bottom-right DP value, i.e. the optimal global alignment score.

## Source

Rodinia benchmark suite, Needleman-Wunsch benchmark.
