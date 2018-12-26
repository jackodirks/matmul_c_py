#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct InformationStruct {
    double* matA;
    double* matB;
    double* matRes;
    size_t size;
    size_t columnCount;
    size_t startColumn;
};

static void matmulNaiveTransposeFDoWork(const double* const restrict a, const double* const restrict b, double* const restrict res, const size_t size, const size_t columnCount, const size_t startColumn){
    printf("%zu, %zu\n", startColumn, startColumn + columnCount - 1);
    for (size_t i = 0; i < size; ++i){
        for (size_t j = startColumn; j < startColumn + columnCount; ++j){
            for (size_t k = 0; k < size; ++k){
                res[i*size + j] += a[i*size + k] * b[j*size + k];
            }
        }
    }
}

static void* matmulNaiveMTTransposeFirstThread(void* s){
    struct InformationStruct* infStruct = s;
    matmulNaiveTransposeFDoWork(infStruct->matA, infStruct->matB, infStruct->matRes, infStruct->size, infStruct->columnCount, infStruct->startColumn);
    return NULL;
}

void matmulMT( double* const restrict a,  double* const restrict b, double* const restrict res, const size_t size, size_t threadCount){
    double* c = malloc(size*size*sizeof(double));
    for (size_t i = 0; i < size; ++i){
        for (size_t j = 0; j < size; ++j){
            c[j + i*size] = b[i + j*size];
        }
    }
    threadCount = size > threadCount ? threadCount : size;
    memset(res, 0, sizeof(double)*size*size);
    pthread_t threads[threadCount - 1];
    struct InformationStruct iStructs[threadCount - 1];
    const size_t columnsPerThread = size/threadCount;
    size_t rest = size % threadCount;
    size_t lastEnd = 0;
    for (size_t i = 0; i < threadCount - 1; ++i){
        iStructs[i].matA = a;
        iStructs[i].matB = c;
        iStructs[i].matRes = res;
        iStructs[i].size = size;
        iStructs[i].columnCount = columnsPerThread;
        iStructs[i].startColumn = lastEnd;
        if (rest > 0){
            iStructs[i].columnCount += 1;
            rest--;
        }
        lastEnd += iStructs[i].columnCount;
        pthread_create(&threads[i], NULL, matmulNaiveMTTransposeFirstThread, (void*)&iStructs[i]);
    }
    matmulNaiveTransposeFDoWork(a, b, res, size, columnsPerThread, lastEnd);
    for (size_t i = 0; i < threadCount - 1; ++i){
        pthread_join(threads[i], NULL);
    }
    free(c);
}
