#include <time.h>
#include <stdlib.h>
#include "matrix_multiply.h"

double now_ms(void)
{
    struct timespec res;
    clock_gettime(CLOCK_REALTIME, &res);
    return 1000.0 * res.tv_sec + (double)res.tv_nsec / 1e6;
}

int main()
{
    double t0, t1, time_c, time_neon;
    float mat1[100][1000];
    float mat2[1000][40];
    float mat2_t[40][1000];
    for (int i = 0; i < 100; ++i)
    {
        for (int j = 0; j < 1000; ++j)
        {
            mat1[i][j] = (float)rand() / ((float)RAND_MAX);
        }
    }
    for (int i = 0; i < 1000; ++i)
    {
        for (int j = 0; j < 40; ++j)
        {
            float temp = (float)rand() / ((float)RAND_MAX);
            mat2[i][j] = temp;
            mat2_t[j][i] = temp;
        }
    }

    float mat3[100][40];
    float mat4[100][40];

    t0 = now_ms();
    for (int i = 0; i < 100; ++i)
    {
        matrix_multiply_naive(100, 1000, 40, &mat1[0][0], &mat2[0][0], &mat3[0][0]);
    }
    time_c = now_ms() - t0;
    printf("naive time: %g ms \n", time_c);

    t1 = now_ms();
    for (int i = 0; i < 100; ++i)
    {
        matrix_multiply_neon(100, 1000, 40, &mat1[0][0], &mat2_t[0][0], &mat4[0][0]);
    }
    time_neon = now_ms() - t1;
    printf("neon time: %g ms \n", time_neon);
    for (int j = 0; j < 4; ++j)
    {
        for (int k = 0; k < 4; ++k)
        {
            printf("%6.2f\n", mat3[j][k] - mat4[j][k]);
        }
    }
    for (int j = 96; j < 100; ++j)
    {
        for (int k = 36; k < 40; ++k)
        {
            printf("%6.2f\n", mat3[j][k] - mat4[j][k]);
        }
    }
    return 0;
}
