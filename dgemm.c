#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <x86intrin.h>
#define ALIGN __attribute__ ((aligned (32)))
#define SIZE 1024
double ALIGN a[SIZE * SIZE];
double ALIGN b[SIZE * SIZE];
double ALIGN c[SIZE * SIZE];
double ALIGN c1[SIZE * SIZE];


// naïve matrix multiplication
void dgemm(int n)
{
    int i, j, k;
    for(i = 0; i < n; i++)
    {
        for(j = 0; j < n; j++)
        {
            double cij = 0;
            for(k = 0; k < n; k++)cij = cij + a[i * n + k] * b[k * n + j];
            c1[i * n + j] = cij;
        }
    }

}

/*********** Loop Unrolling Mechanism ***********/
void dgemm_unrolling(int n)
{
    int i, j, k;
    for(i = 0; i < n; i++)
    {
        for(j = 0; j < n; j++)
        {
            double cij = 0;
            for(k = 0; k < n; k += 4)
            {
                double s1 = a[i * n + k] * b[k * n + j];
                double s2 = a[i * n + (k + 1)] * b[(k + 1) * n + j];
                double s3 = a[i * n + (k + 2)] * b[(k + 2) * n + j];
                double s4 = a[i * n + (k + 3)] * b[(k + 3) * n + j];
                cij += s1 + s2 + s3 + s4;
            }
            c[i * n + j] = cij;
        }
    }
}

/*********** Cache Blocking Mechanism ***********/

#define BLOCK_SIZE 4
void do_block(int n, int si, int sj, int sk, double *a, double *b, double *c)
{
    int i, j, k;
    for (i = si; i < si + BLOCK_SIZE; i++)
        for (j = sj; j < sj + BLOCK_SIZE; j++)
        {
            double cij = c[i * n + j];
            for (k = sk; k < sk + BLOCK_SIZE; k++)
                cij += a[i * n + k] * b[k * n + j];
            c[i * n + j] = cij;
        }
}
void dgemm_blocking(int n)
{
    int i, j, k;
    for(i = 0; i < n; i += BLOCK_SIZE)
        for(j = 0; j < n; j += BLOCK_SIZE)
        {
            c[i * n + j] = 0;
            for(k = 0; k < n; k += BLOCK_SIZE)
                do_block(n, i, j, k, a, b, c);
        }
}

/*********** SIMD Mechanism ***********/
void dgemm_intrin(int n)
{
    int i, j, k;
    for(i = 0; i < n; i += 1)
    {
        for(j = 0; j < n; j += 4)
        {
            __m256d c4  = _mm256_load_pd(&c[i * n + j]);
            for(k = 0; k < n; k++)
            {
                __m256d a4 = _mm256_broadcast_sd(&a[i * n + k]);
                __m256d b4 = _mm256_load_pd(&b[k * n + j]);
                c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
            }
            _mm256_store_pd(&c[i * n + j], c4);
        }
    }
}

/*********** Cache Blocking and SIMD Mechanism ***********/
#define SIMD_BLOCK_SIZE 16
void do_block_SIMD(int n, int si, int sj, int sk, double *a, double *b, double *c)
{
    int i, j, k;
    for (i = si; i < si + SIMD_BLOCK_SIZE; i++)
        for (j = sj; j < sj + SIMD_BLOCK_SIZE; j+=4) {
            __m256d c4  = _mm256_load_pd(&c[i * n + j]);
            for (k = sk; k < sk + SIMD_BLOCK_SIZE; k++) {
                __m256d a4 = _mm256_broadcast_sd(&a[i * n + k]);
                __m256d b4 = _mm256_load_pd(&b[k * n + j]);
                c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
            }
            _mm256_store_pd(&c[i * n + j], c4);
        }
}
void dgemm_blocking_SIMD(int n)
{
    int i, j, k;
    for(i = 0; i < n; i += SIMD_BLOCK_SIZE)
        for(j = 0; j < n; j += SIMD_BLOCK_SIZE)
        {
            c[i * n + j] = 0;
            for(k = 0; k < n; k += SIMD_BLOCK_SIZE)
                do_block_SIMD(n, i, j, k, a, b, c);
        }
}
/*********** Loop Unrolling and SIMD Mechanism ***********/
void dgemm_unrolling_SIMD(int n)
{
    int i, j, k;
    for(i = 0; i < n; i++)
    {
        for(j = 0; j < n; j+=4)
        {
            __m256d c4  = _mm256_load_pd(&c[i * n + j]);
            for(k = 0; k < n; k += 4)   //change here for unrolling tests
            {
              __m256d a4 = _mm256_broadcast_sd(&a[i * n + k]);
              __m256d b4 = _mm256_load_pd(&b[k * n + j]);
              c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
               a4 = _mm256_broadcast_sd(&a[i * n + (k+1)]);
               b4 = _mm256_load_pd(&b[(k+1) * n + j]);
              c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
               a4 = _mm256_broadcast_sd(&a[i * n + (k+2)]);
               b4 = _mm256_load_pd(&b[(k+2) * n + j]);
              c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
               a4 = _mm256_broadcast_sd(&a[i * n + (k+3)]);
               b4 = _mm256_load_pd(&b[(k+3) * n + j]);
              c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
              /**** COMMENT IN/OUT BELOW FOR UNROLLING TESTS ****/
              //  a4 = _mm256_broadcast_sd(&a[i * n + (k+4)]);
              //  b4 = _mm256_load_pd(&b[(k+4) * n + j]);
              // c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
              //  a4 = _mm256_broadcast_sd(&a[i * n + (k+5)]);
              //  b4 = _mm256_load_pd(&b[(k+5) * n + j]);
              // c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
              //  a4 = _mm256_broadcast_sd(&a[i * n + (k+6)]);
              //  b4 = _mm256_load_pd(&b[(k+6) * n + j]);
              // c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
              //  a4 = _mm256_broadcast_sd(&a[i * n + (k+7)]);
              //  b4 = _mm256_load_pd(&b[(k+7) * n + j]);
              // c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
              //  a4 = _mm256_broadcast_sd(&a[i * n + (k+8)]);
              //  b4 = _mm256_load_pd(&b[(k+8) * n + j]);
              // c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
              //  a4 = _mm256_broadcast_sd(&a[i * n + (k+9)]);
              //  b4 = _mm256_load_pd(&b[(k+9) * n + j]);
              // c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
              //  a4 = _mm256_broadcast_sd(&a[i * n + (k+10)]);
              //  b4 = _mm256_load_pd(&b[(k+10) * n + j]);
              // c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
              //  a4 = _mm256_broadcast_sd(&a[i * n + (k+11)]);
              //  b4 = _mm256_load_pd(&b[(k+11) * n + j]);
              // c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
              //  a4 = _mm256_broadcast_sd(&a[i * n + (k+12)]);
              //  b4 = _mm256_load_pd(&b[(k+12) * n + j]);
              // c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
              //  a4 = _mm256_broadcast_sd(&a[i * n + (k+13)]);
              //  b4 = _mm256_load_pd(&b[(k+13) * n + j]);
              // c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
              //  a4 = _mm256_broadcast_sd(&a[i * n + (k+14)]);
              //  b4 = _mm256_load_pd(&b[(k+14) * n + j]);
              // c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
              //  a4 = _mm256_broadcast_sd(&a[i * n + (k+15)]);
              //  b4 = _mm256_load_pd(&b[(k+15) * n + j]);
              // c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
            }
            _mm256_store_pd(&c[i * n + j], c4);
        }
    }
}
/*********** Cache Blocking and Loop Unrolling Mechanism ***********/
#define UNROLL_BLOCK_SIZE 16
void do_block_unrolling(int n, int si, int sj, int sk, double *a, double *b, double *c)
{
    int i, j, k;
    for (i = si; i < si + UNROLL_BLOCK_SIZE; i++)
        for (j = sj; j < sj + UNROLL_BLOCK_SIZE; j++) {
              double cij = c[i * n + j];
            for (k = sk; k < sk + UNROLL_BLOCK_SIZE; k+=4) { //change here for unrolling test
              double s1 = a[i * n + k] * b[k * n + j];
              double s2 = a[i * n + (k + 1)] * b[(k + 1) * n + j];
              double s3 = a[i * n + (k + 2)] * b[(k + 2) * n + j];
              double s4 = a[i * n + (k + 3)] * b[(k + 3) * n + j];
              /**** COMMENT IN/OUT BELOW FOR UNROLLING TESTS ****/
              // double s5 = a[i * n + (k + 4)] * b[(k + 4) * n + j];
              // double s6 = a[i * n + (k + 5)] * b[(k + 5) * n + j];
              // double s7 = a[i * n + (k + 6)] * b[(k + 6) * n + j];
              // double s8 = a[i * n + (k + 7)] * b[(k + 7) * n + j];
              // double s9 = a[i * n + (k + 8)] * b[(k + 8) * n + j];
              // double s10 = a[i * n + (k + 9)] * b[(k + 9) * n + j];
              // double s11 = a[i * n + (k + 10)] * b[(k + 10) * n + j];
              // double s12 = a[i * n + (k + 11)] * b[(k + 11) * n + j];
              // double s13 = a[i * n + (k + 12)] * b[(k + 12) * n + j];
              // double s14 = a[i * n + (k + 13)] * b[(k + 13) * n + j];
              // double s15 = a[i * n + (k + 14)] * b[(k + 14) * n + j];
              // double s16 = a[i * n + (k + 15)] * b[(k + 15) * n + j];
              // cij += s1 + s2;
              cij += s1 + s2 + s3 + s4;
              // cij += s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8;
              // cij += s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10 + s11 + s12 + s13 + s14 + s15 + s16;
            }
            c[i * n + j] = cij;
        }
}
void dgemm_blocking_unrolling(int n)
{
    int i, j, k;
    for(i = 0; i < n; i += UNROLL_BLOCK_SIZE)
        for(j = 0; j < n; j += UNROLL_BLOCK_SIZE)
        {
            c[i * n + j] = 0;
            for(k = 0; k < n; k += UNROLL_BLOCK_SIZE)
                do_block_unrolling(n, i, j, k, a, b, c);
        }
}


/*********** Cache Blocking + Loop Unrolling + SIMD Mechanism ***********/
#define THREE_BLOCK_SIZE 32
void do_block_combined(int n, int si, int sj, int sk, double *a, double *b, double *c)
{
    int i, j, k;
    for (i = si; i < si + THREE_BLOCK_SIZE; i++)
        for (j = sj; j < sj + THREE_BLOCK_SIZE; j+=4) {
              __m256d c4  = _mm256_load_pd(&c[i * n + j]);
            for (k = sk; k < sk + THREE_BLOCK_SIZE; k+=4) { //change here for unrolling tests
              __m256d a4 = _mm256_broadcast_sd(&a[i * n + k]);
              __m256d b4 = _mm256_load_pd(&b[k * n + j]);
              c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
               a4 = _mm256_broadcast_sd(&a[i * n + (k+1)]);
               b4 = _mm256_load_pd(&b[(k+1) * n + j]);
              c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
               a4 = _mm256_broadcast_sd(&a[i * n + (k+2)]);
               b4 = _mm256_load_pd(&b[(k+2) * n + j]);
              c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
               a4 = _mm256_broadcast_sd(&a[i * n + (k+3)]);
               b4 = _mm256_load_pd(&b[(k+3) * n + j]);
              c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4)); 
              /**** COMMENT IN/OUT BELOW FOR UNROLLING TESTS ****/
              //  a4 = _mm256_broadcast_sd(&a[i * n + (k+4)]);
              //  b4 = _mm256_load_pd(&b[(k+4) * n + j]);
              // c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
              //  a4 = _mm256_broadcast_sd(&a[i * n + (k+5)]);
              //  b4 = _mm256_load_pd(&b[(k+5) * n + j]);
              // c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
              //  a4 = _mm256_broadcast_sd(&a[i * n + (k+6)]);
              //  b4 = _mm256_load_pd(&b[(k+6) * n + j]);
              // c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
              //  a4 = _mm256_broadcast_sd(&a[i * n + (k+7)]);
              //  b4 = _mm256_load_pd(&b[(k+7) * n + j]);
              // c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
              //  a4 = _mm256_broadcast_sd(&a[i * n + (k+8)]);
              //  b4 = _mm256_load_pd(&b[(k+8) * n + j]);
              // c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
              //  a4 = _mm256_broadcast_sd(&a[i * n + (k+9)]);
              //  b4 = _mm256_load_pd(&b[(k+9) * n + j]);
              // c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
              //  a4 = _mm256_broadcast_sd(&a[i * n + (k+10)]);
              //  b4 = _mm256_load_pd(&b[(k+10) * n + j]);
              // c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
              //  a4 = _mm256_broadcast_sd(&a[i * n + (k+11)]);
              //  b4 = _mm256_load_pd(&b[(k+11) * n + j]);
              // c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
              //  a4 = _mm256_broadcast_sd(&a[i * n + (k+12)]);
              //  b4 = _mm256_load_pd(&b[(k+12) * n + j]);
              // c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
              //  a4 = _mm256_broadcast_sd(&a[i * n + (k+13)]);
              //  b4 = _mm256_load_pd(&b[(k+13) * n + j]);
              // c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
              //  a4 = _mm256_broadcast_sd(&a[i * n + (k+14)]);
              //  b4 = _mm256_load_pd(&b[(k+14) * n + j]);
              // c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
              //  a4 = _mm256_broadcast_sd(&a[i * n + (k+15)]);
              //  b4 = _mm256_load_pd(&b[(k+15) * n + j]);
              // c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));

            }
            _mm256_store_pd(&c[i * n + j], c4);
        }
}
void dgemm_blocking_combined(int n)
{
    int i, j, k;
    for(i = 0; i < n; i += THREE_BLOCK_SIZE)
        for(j = 0; j < n; j += THREE_BLOCK_SIZE)
        {
            c[i * n + j] = 0;
            for(k = 0; k < n; k += THREE_BLOCK_SIZE)
                do_block_combined(n, i, j, k, a, b, c);
        }
}


/* Implement this function with multiple optimization techniques. */
void optimized_dgemm(int n)
{
    // call any of optimization attempt
    //dgemm_blocking_combined(n); //All three mechanisms
    dgemm_blocking_unrolling(n); //Cache Blocking + Loop Unrolling
    //dgemm_unrolling_SIMD(n); //Loop Unrolling + SIMD
    //dgemm_blocking_SIMD(n); //Cache Blocking + SIMD
    //dgemm_intrin(n); //SIMD
    //dgemm_blocking(n); //Cache Blocking
}

void main(int argc, char **argv)
{
    int i, j;
    time_t t;
    struct timeval start, end;
    double elapsed_time;
    int check_correctness = 0;
    int correct = 1;

    if(argc > 1)
    {
        if(strcmp(argv[1], "corr") == 0)
        {
            check_correctness = 1;
        }
    }
    /* Initialize random number generator */
    srand((unsigned) time(&t));
    /* Populate the arrays with random values */
    for(i = 0; i < SIZE; i++)
    {
        for(j = 0; j < SIZE; j++)
        {
            a[i * SIZE + j] = (double)rand() / (RAND_MAX + 1.0);
            b[i * SIZE + j] = (double)rand() / (RAND_MAX + 1.0);
            c[i * SIZE + j] = 0.0;
            c1[i * SIZE + j] = 0.0;
        }
    }
    gettimeofday(&start, NULL);
    /* Call you optimized function optimized_dgemm */
    optimized_dgemm(SIZE);
    gettimeofday(&end, NULL);
    /* For TA use only */
    if(check_correctness)
    {
        dgemm(SIZE);
        for(i = 0; (i < SIZE) && correct ; i++)
        {
            for(j = 0; (j < SIZE) && correct; j++)
            {
                if(fabs(c[i * SIZE + j] - c1[i * SIZE + j]) >= 0.0000001)
                {
                    printf("%f != %f\n", c[i * SIZE + j], c1[i * SIZE + j]);
                    correct = 0;
                }
            }
        }
        if(correct) printf("Result is correct!\n");
        else printf("Result is incorrect!\n");
    }
    elapsed_time = (end.tv_sec - start.tv_sec) * 1000.0;
    elapsed_time += (end.tv_usec - start.tv_usec) / 1000.0;
    printf("dgemm finished in %f milliseconds.\n", elapsed_time);
}
