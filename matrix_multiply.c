#include "matrix_multiply.h"
#include <arm_neon.h>

void matrix_multiply_naive(int x, int y, int z,
                           float *m1, float *m2, float *r
                           ) {
    float *m1p;
    float *m2p;
    int k;
    int row;
    int column;
    for (row = 0; row < x; row += 1) {
        m1p = m1 + row * y;
        for (column = 0; column < z; column += 1) {
            m2p = m2 + column;
            *r = 0;
            for (k = 0; k < y; k += 1) {
                *r += m1p[k] * m2p[k * z];
            }
            r++;
        }
    }
}

void matrix_multiply_neon(int m, int n, int p, float *mat1, float *mat2, float *mat3) {
    // REQUIRE n mod 4 == 0
    float32x4_t sum, a, x;
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < p; ++k) {
            sum = vdupq_n_f32(0.f);
            for (int j = 0; j < n; j += 4) {
                a = vld1q_f32(mat1 + i * n + j);
                x = vld1q_f32(mat2 + k * n + j);
                sum = vmlaq_f32(sum, a, x);
            }
            mat3[i * p + k] = vgetq_lane_f32(sum, 0)
                            + vgetq_lane_f32(sum, 1)
                            + vgetq_lane_f32(sum, 2)
                            + vgetq_lane_f32(sum, 3);
        }
    }
}
