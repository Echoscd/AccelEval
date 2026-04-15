#include <stddef.h>
static int g_n = 0; static const int *g_row_ptr = 0; static const int *g_col_idx = 0; static const double *g_values = 0; static const double *g_x = 0;
void solution_init(int n, const int *row_ptr, const int *col_idx, const double *values, const double *x) { g_n=n; g_row_ptr=row_ptr; g_col_idx=col_idx; g_values=values; g_x=x; }
void solution_compute(double *y_out) { for (int i=0;i<g_n;++i) { double sum=0.0; for (int j=g_row_ptr[i]; j<g_row_ptr[i+1]; ++j) sum += g_values[j]*g_x[g_col_idx[j]]; y_out[i]=sum; } }
void solution_free(void) { g_n=0; g_row_ptr=0; g_col_idx=0; g_values=0; g_x=0; }
