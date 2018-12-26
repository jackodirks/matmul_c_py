#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

static double dotProduct_4(const double* const restrict u, const double* const restrict v){
    __m256d dp = _mm256_mul_pd(_mm256_load_pd(u), _mm256_load_pd(v));
    __m128d a = _mm256_extractf128_pd(dp, 0);
    __m128d b = _mm256_extractf128_pd(dp, 1);
    __m128d c = _mm_add_pd(a, b);
    __m128d yy = _mm_unpackhi_pd(c, c);
    __m128d dotproduct  = _mm_add_pd(c, yy);
    return _mm_cvtsd_f64(dotproduct);
}

static __m256d fourDotProductsFour(double* restrict u0, double* restrict v0, double* restrict u1, double* restrict v1){
    __m256d xy0 = _mm256_mul_pd(_mm256_load_pd(u0), _mm256_load_pd(v0));
    __m256d xy1 = _mm256_mul_pd(_mm256_load_pd(u1), _mm256_load_pd(v1));
    __m256d xy2 = _mm256_mul_pd(_mm256_load_pd(u1), _mm256_load_pd(v0));
    __m256d xy3 = _mm256_mul_pd(_mm256_load_pd(u0), _mm256_load_pd(v1));
    __m256d temp01 = _mm256_hadd_pd(xy0, xy1);
    __m256d temp23 = _mm256_hadd_pd(xy2, xy3);
    __m256d swapped = _mm256_permute2f128_pd(temp01, temp23, 0x21);
    __m256d blended = _mm256_blend_pd(temp01, temp23, 0b1100);
    return _mm256_add_pd(swapped, blended);
}

static uintptr_t offset32Alignment(const double* const ptr){
    uintptr_t val = (uintptr_t)ptr;
    return val % 32;
}

void simdMultiplyFour(double* restrict a, double* const restrict b, double* const restrict res, size_t size){
    if (size % 4 != 0){
        fprintf(stderr, "Size is not a multiple of 4\n");
        exit(1);
    }
    memset(res, 0, sizeof(double)*size*size);
    double* c = aligned_alloc(32, size*size*sizeof(double));
    if(offset32Alignment(a) != 0){
        fprintf(stderr, "Array a is not correctly aligned\n");
        exit(1);
    }
    for (size_t i = 0; i < size; ++i){
        for (size_t j = 0; j < size; ++j){
            c[j + i*size] = b[i + j*size];
        }
    }

    for (size_t i = 0; i < size; ++i){
        for (size_t j = 0; j < size; ++j){
            for (size_t k = 0; k < size; k+=4){
                res[i*size + j] += dotProduct_4(&a[i*size + k],  &c[j*size + k]);
            }
        }
    }
    free(c);
}

void simdMoreOptimized(double* const restrict a, double* const restrict b, double* const restrict res, const size_t size){
    if (size % 4 != 0){
        fprintf(stderr, "Size is not a multiple of 4\n");
        exit(1);
    }
    memset(res, 0, sizeof(double)*size*size);
    double* c = aligned_alloc(32, size*size*sizeof(double));
    if(offset32Alignment(a) != 0){
        fprintf(stderr, "Array a is not correctly aligned\n");
        exit(1);
    }
    for (size_t i = 0; i < size; ++i){
        for (size_t j = 0; j < size; ++j){
            c[j + i*size] = b[i + j*size];
        }
    }
    for (size_t i = 0; i < size; i+=2){
        for (size_t j = 0; j < size; j+=2){
            for (size_t k = 0; k < size; k+=4){
                __m256d t = fourDotProductsFour(&a[i*size + k], &c[j*size + k], &a[(i+1)*size + k], &c[(j+1)*size + k]);
                double* temp = (double*)&t;
                res[i*size + j] += temp[0];
                res[(i+1)*size + (j+1)] += temp[1];
                res[(i+1)*size + j] += temp[2];
                res[i*size + (j+1)] += temp[3];
            }
        }
    }
    free(c);
}
