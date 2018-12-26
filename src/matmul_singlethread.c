#include <stdlib.h>
#include <stdio.h>

void matmulNaive(const double* const restrict a, const double* const restrict b, double* const restrict res, const size_t size){
	for (size_t i = 0; i < size; ++i){
		for (size_t j = 0; j < size; ++j){
			res[i*size + j] = 0;
			for (size_t k = 0; k < size; ++k){
				res[i*size + j] += a[i*size + k] * b[k*size + j];
			}
		}
	}
}

void matmulNaiveTransposeFirst(const double* const restrict a, double* const restrict b, double* const restrict res, const size_t size){
	double* c = malloc(size*size*sizeof(double));
	for (size_t i = 0; i < size; ++i){
		for (size_t j = 0; j < size; ++j){
            printf("%f\n", b[i + j*size]);
			c[j + i*size] = b[i + j*size];
			res[i*size + j] = 0;
		}
	}
	for (size_t i = 0; i < size; ++i){
		for (size_t j = 0; j < size; ++j){
			for (size_t k = 0; k < size; ++k){
				res[i*size + j] += a[i*size + k] * c[j*size + k];
			}
		}
	}
	free(c);
}

void matmulNaiveBlock(const int32_t* const restrict a, int32_t* const restrict b, int32_t* const restrict res, const size_t size){
	for (size_t i = 0; i < size; ++i){
		for (size_t j = 0; j < size; ++j){
			res[i*size + j] = 0;
			for (size_t k = 0; k < size; ++k){
				res[i*size + j] += a[i*size + k] * b[k*size + j];
			}
		}
	}
}

