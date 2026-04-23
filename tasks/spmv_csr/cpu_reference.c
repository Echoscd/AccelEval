// cpu_reference.c — Sparse Matrix-Vector Multiply: result = A^T * vector
//
// Verbatim sequential port of Google OR-Tools PDLP:
//   ortools/pdlp/sharder.cc  lines 160–173
//   TransposedMatrixVectorProduct(matrix, vector, sharder)
//
// Original C++ (Eigen + Sharder parallelism):
//   VectorXd TransposedMatrixVectorProduct(
//       const SparseMatrix<double, ColMajor, int64_t>& matrix,
//       const VectorXd& vector, const Sharder& sharder) {
//     CHECK_EQ(vector.size(), matrix.rows());
//     VectorXd answer(matrix.cols());
//     sharder.ParallelForEachShard([&](const Sharder::Shard& shard) {
//       shard(answer) = shard(matrix).transpose() * vector;
//     });
//     return answer;
//   }
//
// The Sharder partitions columns into contiguous shards.  Within each shard
// the Eigen expression `shard(matrix).transpose() * vector` computes, for
// every column j in the shard:
//     answer[j] = sum_{k in [col_ptrs[j], col_ptrs[j+1])} values[k] * vector[row_indices[k]]
// i.e. the dot product of column j of A with the dense vector.
//
// This CPU port removes the Sharder parallelism and executes the column-
// gather dot product sequentially over all columns, preserving the original
// CSC (ColMajor) storage layout and the identical arithmetic.
//
// Data layout (matching Eigen::SparseMatrix<float, ColMajor>):
//   col_ptrs   [num_cols + 1]  — column pointer array
//   row_indices [nnz]          — row index of each nonzero
//   values      [nnz]          — value of each nonzero
//   vector      [num_rows]     — dense input
//   answer      [num_cols]     — dense output
//
// NO file I/O, NO main(). All I/O handled by task_io_cpu.c.

#include <stddef.h>

// ===== Module-level state =====
static int          g_num_rows;   // matrix.rows()
static int          g_num_cols;   // matrix.cols()
static const int*   g_col_ptrs;   // CSC outer index [num_cols + 1]
static const int*   g_row_indices;// CSC inner index [nnz]
static const float* g_values;     // CSC values [nnz]
static const float* g_vector;     // dense input [num_rows]

// ===== TransposedMatrixVectorProduct — sequential single-shard port =====
// answer[j] = shard(matrix).transpose() * vector  for column j
//           = dot(column_j_of_A, vector)
//           = sum_{k in [col_ptrs[j], col_ptrs[j+1])} values[k] * vector[row_indices[k]]
static void TransposedMatrixVectorProduct(
    int          num_cols,
    const int*   col_ptrs,
    const int*   row_indices,
    const float* values,
    const float* vector,
    float*       answer)
{
    int j, k;
    for (j = 0; j < num_cols; j++) {
        float sum = 0.0f;
        for (k = col_ptrs[j]; k < col_ptrs[j + 1]; k++) {
            sum += values[k] * vector[row_indices[k]];
        }
        answer[j] = sum;
    }
}

// ===== Public interface =====

static void _orbench_old_init(int          num_rows,
                   int          num_cols,
                   const int*   col_ptrs,
                   const int*   row_indices,
                   const float* values,
                   const float* vector)
{
    g_num_rows    = num_rows;
    g_num_cols    = num_cols;
    g_col_ptrs    = col_ptrs;
    g_row_indices = row_indices;
    g_values      = values;
    g_vector      = vector;
}

static void _orbench_old_compute(int num_cols, float* answer)
{
    TransposedMatrixVectorProduct(num_cols, g_col_ptrs, g_row_indices,
                                 g_values, g_vector, answer);
}

// ── Unified compute_only wrapper (auto-migrated) ──
void solution_compute(int num_rows, int num_cols, const int* col_ptrs, const int* row_indices, const float* values, const float* vector, float* answer) {
    _orbench_old_init(num_rows, num_cols, col_ptrs, row_indices, values, vector);
    _orbench_old_compute(num_cols, answer);
}
