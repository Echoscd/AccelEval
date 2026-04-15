#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define BS 16

static int g_n = 0;
static const float* g_input = NULL;

#define AA(mat, offset, size, i, j) ((mat)[(size_t)((offset)+(i)) * (size) + ((offset)+(j))])
#define BB(mat, size, i, j) ((mat)[(size_t)(i) * (size) + (j)])

static void lud_diagonal(float* a, int size, int offset) {
    for (int i = 0; i < BS; i++) {
        for (int j = i; j < BS; j++) {
            for (int k = 0; k < i; k++) {
                AA(a, offset, size, i, j) -= AA(a, offset, size, i, k) * AA(a, offset, size, k, j);
            }
        }
        float temp = 1.0f / AA(a, offset, size, i, i);
        for (int j = i + 1; j < BS; j++) {
            for (int k = 0; k < i; k++) {
                AA(a, offset, size, j, i) -= AA(a, offset, size, j, k) * AA(a, offset, size, k, i);
            }
            AA(a, offset, size, j, i) *= temp;
        }
    }
}

static void lud_blocked(float* a, int size) {
    for (int offset = 0; offset < size - BS; offset += BS) {
        int size_inter = size - offset - BS;
        int chunks_in_inter_row = size_inter / BS;

        for (int chunk_idx = 0; chunk_idx < chunks_in_inter_row; chunk_idx++) {
            int i_global, j_global, i_here, j_here;
            float sum;
            float temp[BS * BS];

            for (int i = 0; i < BS; i++) {
                for (int j = 0; j < BS; j++) {
                    temp[i * BS + j] = a[(size_t)(i + offset) * size + offset + j];
                }
            }

            i_global = offset;
            j_global = offset + BS * (chunk_idx + 1);
            for (int j = 0; j < BS; j++) {
                for (int i = 0; i < BS; i++) {
                    sum = 0.0f;
                    for (int k = 0; k < i; k++) {
                        sum += temp[BS * i + k] * BB(a, size, i_global + k, j_global + j);
                    }
                    i_here = i_global + i;
                    j_here = j_global + j;
                    BB(a, size, i_here, j_here) -= sum;
                }
            }

            j_global = offset;
            i_global = offset + BS * (chunk_idx + 1);
            for (int i = 0; i < BS; i++) {
                for (int j = 0; j < BS; j++) {
                    sum = 0.0f;
                    for (int k = 0; k < j; k++) {
                        sum += BB(a, size, i_global + i, j_global + k) * temp[BS * k + j];
                    }
                    i_here = i_global + i;
                    j_here = j_global + j;
                    a[(size_t)i_here * size + j_here] = (a[(size_t)i_here * size + j_here] - sum) /
                                                       a[(size_t)(offset + j) * size + offset + j];
                }
            }
        }

        int chunks_per_inter = chunks_in_inter_row * chunks_in_inter_row;
        for (int chunk_idx = 0; chunk_idx < chunks_per_inter; chunk_idx++) {
            int i_global = offset + BS * (1 + chunk_idx / chunks_in_inter_row);
            int j_global = offset + BS * (1 + chunk_idx % chunks_in_inter_row);
            float temp_top[BS * BS];
            float temp_left[BS * BS];
            float sum[BS];
            for (int j = 0; j < BS; j++) sum[j] = 0.0f;

            for (int i = 0; i < BS; i++) {
                for (int j = 0; j < BS; j++) {
                    temp_top[i * BS + j] = a[(size_t)(i + offset) * size + j + j_global];
                    temp_left[i * BS + j] = a[(size_t)(i + i_global) * size + offset + j];
                }
            }

            for (int i = 0; i < BS; i++) {
                for (int k = 0; k < BS; k++) {
                    for (int j = 0; j < BS; j++) {
                        sum[j] += temp_left[BS * i + k] * temp_top[BS * k + j];
                    }
                }
                for (int j = 0; j < BS; j++) {
                    BB(a, size, i + i_global, j + j_global) -= sum[j];
                    sum[j] = 0.0f;
                }
            }
        }

        lud_diagonal(a, size, offset);
    }

    lud_diagonal(a, size, size - BS);
}

void solution_init(int n, const float* A0) {
    g_n = n;
    g_input = A0;
}

void solution_compute(float* LU) {
    if (!LU || !g_input || g_n <= 0) return;
    memcpy(LU, g_input, (size_t)g_n * (size_t)g_n * sizeof(float));
    lud_blocked(LU, g_n);
}

void solution_free(void) {
    g_n = 0;
    g_input = NULL;
}
