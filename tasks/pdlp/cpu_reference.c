// cpu_reference.c — PDHG LP Solver Iterations (CPU baseline)
//
// Faithfully ported from Google OR-Tools PDLP solver:
//   ortools/pdlp/primal_dual_hybrid_gradient.cc
//     — TakeConstantSizeStep()           [lines 2644–2675]
//     — ComputeNextPrimalSolution()      [lines 1834–1880, LP branch]
//     — ComputeNextDualSolution()        [lines 1882–1933]
//   ortools/pdlp/sharder.cc
//     — TransposedMatrixVectorProduct()  [lines 160–173]
//   ortools/pdlp/sharded_optimization_utils.cc
//     — ShardedWeightedAverage::Add()    [lines 54–66]
//
// Reference: Applegate, Díaz, Hinder, Lu, Lubin, O'Donoghue & Woodruff,
// "Practical Large-Scale Linear Programming using Primal-Dual Hybrid Gradient",
// NeurIPS 2021 (https://arxiv.org/abs/2106.04756).
// Algorithm follows Chambolle & Pock (2011, 2014).
//
// Data layout:
//   Constraint matrix A stored CSC (Compressed Sparse Column) matching
//   Eigen::SparseMatrix<float, ColMajor>. Float32 throughout.
//
// NO file I/O, NO main(). All I/O handled by task_io_cpu.c.

#include <math.h>
#include <stdlib.h>
#include <string.h>

// ===== Module-level state =====
static int   g_num_vars;
static int   g_num_constraints;
static int   g_nnz;
static int   g_num_iters;
static float g_step_size;
static float g_primal_weight;

static const float* g_obj;          // c [num_vars]
static const float* g_var_lb;       // l_x [num_vars]
static const float* g_var_ub;       // u_x [num_vars]
static const float* g_con_lb;       // l_c [num_constraints]
static const float* g_con_ub;       // u_c [num_constraints]
static const int*   g_col_ptrs;     // CSC [num_vars + 1]
static const int*   g_row_indices;  // CSC [nnz]
static const float* g_values;       // CSC [nnz]

// Working buffers
static float* g_primal;           // x [num_vars]
static float* g_dual;             // y [num_constraints]
static float* g_dual_product;     // A^T y [num_vars]
static float* g_primal_avg;       // running average of x [num_vars]
static float  g_avg_weight_sum;

// ===== SpMV: A^T * y (transpose multiply, CSC natural direction) =====
// Faithfully matches TransposedMatrixVectorProduct() in sharder.cc:160–173
// which computes shard(A).transpose() * vector per column-shard.
static void spmv_ATy(int num_vars, const int* col_ptrs,
                     const int* row_indices, const float* values,
                     const float* y, float* result)
{
    int j, k;
    for (j = 0; j < num_vars; j++) {
        float sum = 0.0f;
        for (k = col_ptrs[j]; k < col_ptrs[j + 1]; k++) {
            sum += values[k] * y[row_indices[k]];
        }
        result[j] = sum;
    }
}

// ===== SpMV: A * x (forward multiply, CSC scattered) =====
// Used internally in ComputeNextDualSolution via
// shard(TransposedConstraintMatrix).transpose() * extrapolated_primal
// (primal_dual_hybrid_gradient.cc:1914–1916)
static void spmv_Ax(int num_vars, int num_constraints,
                    const int* col_ptrs, const int* row_indices,
                    const float* values, const float* x, float* result)
{
    int j, k;
    memset(result, 0, (size_t)num_constraints * sizeof(float));
    for (j = 0; j < num_vars; j++) {
        float xj = x[j];
        for (k = col_ptrs[j]; k < col_ptrs[j + 1]; k++) {
            result[row_indices[k]] += values[k] * xj;
        }
    }
}

// ===== One constant-size PDHG step =====
// Faithful port of Solver::TakeConstantSizeStep() (lines 2644–2675)
// calling ComputeNextPrimalSolution (LP branch, lines 1867–1874)
// and ComputeNextDualSolution (no Malitsky-Pock, lines 1893–1931).
static void pdhg_step(int   num_vars,
                      int   num_constraints,
                      float step_size,
                      float primal_weight,
                      const float* obj,
                      const float* var_lb, const float* var_ub,
                      const float* con_lb, const float* con_ub,
                      const int*   col_ptrs,
                      const int*   row_indices,
                      const float* values,
                      float* primal,       // x, updated in place
                      float* dual,         // y, updated in place
                      float* dual_product, // A^T y, updated in place
                      float* primal_avg,   // running average, updated
                      float* avg_weight_sum,
                      float* tmp_x_bar,    // scratch [num_vars]
                      float* tmp_Ax,       // scratch [num_constraints]
                      float* tmp_temp)     // scratch [num_constraints]
{
    int j, i;
    // Step sizes (primal_dual_hybrid_gradient.cc:2645–2646)
    const float primal_step_size = step_size / primal_weight;
    const float dual_step_size   = step_size * primal_weight;

    // --- ComputeNextPrimalSolution (LP branch, lines 1867–1874) ---
    // x_new[j] = clip(x[j] - τ*(c[j] - dual_product[j]), var_lb[j], var_ub[j])
    // x_bar[j] = 2*x_new[j] - x[j]  (extrapolation_factor = 1.0, line 2650)
    for (j = 0; j < num_vars; j++) {
        float x_old = primal[j];
        float grad = obj[j] - dual_product[j];
        float x_new = x_old - primal_step_size * grad;
        // clip to variable bounds (cwiseMin/cwiseMax in source)
        if (x_new > var_ub[j]) x_new = var_ub[j];
        if (x_new < var_lb[j]) x_new = var_lb[j];
        primal[j] = x_new;
        // Extrapolated primal (lines 1895–1900): x_bar = x_new + 1*(x_new - x_old)
        tmp_x_bar[j] = 2.0f * x_new - x_old;
    }

    // --- ComputeNextDualSolution (lines 1902–1931) ---
    // SpMV: A * x_bar → tmp_Ax  (line 1914: shard(TransposedConstraintMatrix).transpose() * extrapolated_primal)
    spmv_Ax(num_vars, num_constraints, col_ptrs, row_indices, values,
            tmp_x_bar, tmp_Ax);

    // temp = y - σ * A * x_bar  (line 1912–1916)
    for (i = 0; i < num_constraints; i++) {
        tmp_temp[i] = dual[i] - dual_step_size * tmp_Ax[i];
    }

    // Dual projection (lines 1923–1928):
    // y_new = max(temp + σ*con_lb, min(0, temp + σ*con_ub))
    for (i = 0; i < num_constraints; i++) {
        float v_ub = tmp_temp[i] + dual_step_size * con_ub[i];
        float v_lb = tmp_temp[i] + dual_step_size * con_lb[i];
        float y_new = 0.0f;
        if (y_new > v_ub) y_new = v_ub;  // cwiseMin with 0
        if (y_new < v_lb) y_new = v_lb;  // cwiseMax with lower
        dual[i] = y_new;
    }

    // --- Update dual_product = A^T * y_new (line 2663–2665) ---
    spmv_ATy(num_vars, col_ptrs, row_indices, values, dual, dual_product);

    // --- Weighted average update (sharded_optimization_utils.cc:54–66) ---
    // M_14 algorithm: avg += (w / (sum_w + w)) * (x - avg)
    {
        float w = step_size;
        float ratio = w / (*avg_weight_sum + w);
        for (j = 0; j < num_vars; j++) {
            primal_avg[j] += ratio * (primal[j] - primal_avg[j]);
        }
        *avg_weight_sum += w;
    }
}

// ===== Public interface =====

static void _orbench_old_init(int          num_vars,
                   int          num_constraints,
                   int          nnz,
                   int          num_iters,
                   const float* obj,
                   const float* var_lb,
                   const float* var_ub,
                   const float* con_lb,
                   const float* con_ub,
                   const int*   col_ptrs,
                   const int*   row_indices,
                   const float* values,
                   float        step_size,
                   float        primal_weight)
{
    g_num_vars        = num_vars;
    g_num_constraints = num_constraints;
    g_nnz             = nnz;
    g_num_iters       = num_iters;
    g_step_size       = step_size;
    g_primal_weight   = primal_weight;
    g_obj             = obj;
    g_var_lb          = var_lb;
    g_var_ub          = var_ub;
    g_con_lb          = con_lb;
    g_con_ub          = con_ub;
    g_col_ptrs        = col_ptrs;
    g_row_indices     = row_indices;
    g_values          = values;

    // Allocate working buffers (zero-initialized, matching PDLP's default start)
    g_primal       = (float*)calloc(num_vars, sizeof(float));
    g_dual         = (float*)calloc(num_constraints, sizeof(float));
    g_dual_product = (float*)calloc(num_vars, sizeof(float));
    g_primal_avg   = (float*)calloc(num_vars, sizeof(float));
    g_avg_weight_sum = 0.0f;
}

static void _orbench_old_compute(int num_vars, int num_constraints, float* primal_out)
{
    // Scratch buffers for one step
    float* tmp_x_bar = (float*)malloc(g_num_vars * sizeof(float));
    float* tmp_Ax    = (float*)malloc(g_num_constraints * sizeof(float));
    float* tmp_temp  = (float*)malloc(g_num_constraints * sizeof(float));

    // Reset state (idempotent: matches PDLP starting from zero)
    memset(g_primal,       0, g_num_vars * sizeof(float));
    memset(g_dual,         0, g_num_constraints * sizeof(float));
    memset(g_dual_product, 0, g_num_vars * sizeof(float));
    memset(g_primal_avg,   0, g_num_vars * sizeof(float));
    g_avg_weight_sum = 0.0f;

    // Run K iterations of constant-size PDHG (TakeConstantSizeStep)
    for (int iter = 0; iter < g_num_iters; iter++) {
        pdhg_step(g_num_vars, g_num_constraints,
                  g_step_size, g_primal_weight,
                  g_obj, g_var_lb, g_var_ub, g_con_lb, g_con_ub,
                  g_col_ptrs, g_row_indices, g_values,
                  g_primal, g_dual, g_dual_product,
                  g_primal_avg, &g_avg_weight_sum,
                  tmp_x_bar, tmp_Ax, tmp_temp);
    }

    // Output the averaged primal solution (matching PDLP's PrimalAverage())
    memcpy(primal_out, g_primal_avg, g_num_vars * sizeof(float));

    free(tmp_x_bar);
    free(tmp_Ax);
    free(tmp_temp);
}

// ── Unified compute_only wrapper (auto-migrated) ──
void solution_compute(int num_vars, int num_constraints, int nnz, int num_iters, const float* obj, const float* var_lb, const float* var_ub, const float* con_lb, const float* con_ub, const int* col_ptrs, const int* row_indices, const float* values, float step_size, float primal_weight, float* primal_out) {
    _orbench_old_init(num_vars, num_constraints, nnz, num_iters, obj, var_lb, var_ub, con_lb, con_ub, col_ptrs, row_indices, values, step_size, primal_weight);
    _orbench_old_compute(num_vars, num_constraints, primal_out);
}
