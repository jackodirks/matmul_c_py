#include <immintrin.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <malloc.h>
struct InformationStruct {
    double* matA;
    double* matB;
    double* matRes;
    size_t size;
    size_t columnCount;
    size_t startColumn;
};

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


static void worker(double* const restrict a, double* const restrict b, double* const restrict res, const size_t size, const size_t columnCount, const size_t startColumn){
    for (size_t i = 0; i < size; i+=2){
        for (size_t j = startColumn; j < startColumn + columnCount; j+=2){
            for (size_t k = 0; k < size; k+=4){
                __m256d t = fourDotProductsFour(&a[i*size + k], &b[j*size + k], &a[(i+1)*size + k], &b[(j+1)*size + k]);
                double* temp = (double*)&t;
                res[i*size + j] += temp[0];
                res[(i+1)*size + (j+1)] += temp[1];
                res[(i+1)*size + j] += temp[2];
                res[i*size + (j+1)] += temp[3];
            }
        }
    }
}

static void* thread(void* s){
    struct InformationStruct* infStruct = s;
    worker(infStruct->matA, infStruct->matB, infStruct->matRes, infStruct->size, infStruct->columnCount, infStruct->startColumn);
    return NULL;
}

void matmulSIMDMT(double* const restrict a,  double* const restrict b, double* const restrict res, const size_t size, size_t threadCount){
    if (size % 4 != 0){
        fprintf(stderr, "Size is not a multiple of 4\n");
        exit(1);
    }
    double* c = aligned_alloc(32, size*size*sizeof(double));
    for (size_t i = 0; i < size; ++i){
        for (size_t j = 0; j < size; ++j){
            c[j + i*size] = b[i + j*size];
        }
    }
    if(offset32Alignment(a) != 0){
        fprintf(stderr, "Array a is not correctly aligned\n");
        exit(1);
    }
    threadCount = size/2 > threadCount ? threadCount : size/2;
    memset(res, 0, sizeof(double)*size*size);
    pthread_t threads[threadCount - 1];
    struct InformationStruct iStructs[threadCount - 1];
    const size_t columnsPerThread = ((size/2)/threadCount)*2;
    size_t rest = ((size/2) % threadCount)*2;
    size_t lastEnd = 0;
    for (size_t i = 0; i < threadCount - 1; ++i){
        iStructs[i].matA = a;
        iStructs[i].matB = c;
        iStructs[i].matRes = res;
        iStructs[i].size = size;
        iStructs[i].columnCount = columnsPerThread;
        iStructs[i].startColumn = lastEnd;
        if (rest > 0){
            iStructs[i].columnCount += 2;
            rest-=2;
        }
        lastEnd += iStructs[i].columnCount;
        pthread_create(&threads[i], NULL, thread, (void*)&iStructs[i]);
    }
    worker(a, c, res, size, columnsPerThread, lastEnd);
    for (size_t i = 0; i < threadCount - 1; ++i){
        pthread_join(threads[i], NULL);
    }
    free(c);
}
