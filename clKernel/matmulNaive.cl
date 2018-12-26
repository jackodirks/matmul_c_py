#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void matmul(__global const double *a, __global const double *b, __global double *res, unsigned int size) {
    // Get the index of the current element to be processed
    size_t row = get_global_id(0);
    size_t col = get_global_id(1);
    double ans = 0;
    for (size_t i = 0; i < size; ++i){
         ans += a[row*size + i] * b[col*size + i];
    }
    res[row*size + col] = ans;
}

