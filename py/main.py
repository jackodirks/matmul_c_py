#!/usr/bin/python
import ctypes
import numpy
import timeit
import sys

matmullib = ctypes.CDLL('./libmatmul')
matmullib.sortList.argtypes = [numpy.ctypeslib.ndpointer(dtype=numpy.int32, flags = ["C_CONTIGUOUS", "ALIGNED"]), ctypes.c_size_t]
matmullib.matmulNaive.argtypes = [numpy.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=2, flags = ["C_CONTIGUOUS", "ALIGNED"]),
                            numpy.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim = 2, flags = ["C_CONTIGUOUS", "ALIGNED"]),
                            numpy.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim = 2, flags = ["C_CONTIGUOUS", "ALIGNED"]),
                            ctypes.c_size_t]
matmullib.matmulNaiveTransposeFirst.argtypes = [numpy.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=2, flags = ["C_CONTIGUOUS", "ALIGNED"]),
                            numpy.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim = 2, flags = ["C_CONTIGUOUS", "ALIGNED"]),
                            numpy.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim = 2, flags = ["C_CONTIGUOUS", "ALIGNED"]),
                            ctypes.c_size_t]
matmullib.matmulMT.argtypes = [numpy.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=2, flags = ["C_CONTIGUOUS", "ALIGNED"]),
                            numpy.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim = 2, flags = ["C_CONTIGUOUS", "ALIGNED"]),
                            numpy.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim = 2, flags = ["C_CONTIGUOUS", "ALIGNED"]),
                            ctypes.c_size_t,
                            ctypes.c_size_t]
matmullib.matmulOpenClNaive.argtypes = [numpy.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=2, flags = ["C_CONTIGUOUS", "ALIGNED"]),
                            numpy.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim = 2, flags = ["C_CONTIGUOUS", "ALIGNED"]),
                            numpy.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim = 2, flags = ["C_CONTIGUOUS", "ALIGNED"]),
                            ctypes.c_size_t]
matmullib.simdMultiplyFour.argtypes = [numpy.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=2, flags = ["C_CONTIGUOUS", "ALIGNED"]),
                            numpy.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim = 2, flags = ["C_CONTIGUOUS", "ALIGNED"]),
                            numpy.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim = 2, flags = ["C_CONTIGUOUS", "ALIGNED"]),
                            ctypes.c_size_t]
matmullib.simdMoreOptimized.argtypes = [numpy.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=2, flags = ["C_CONTIGUOUS", "ALIGNED"]),
                            numpy.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim = 2, flags = ["C_CONTIGUOUS", "ALIGNED"]),
                            numpy.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim = 2, flags = ["C_CONTIGUOUS", "ALIGNED"]),
                            ctypes.c_size_t]
matmullib.matmulSIMDMT.argtypes = [numpy.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=2, flags = ["C_CONTIGUOUS", "ALIGNED"]),
                            numpy.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim = 2, flags = ["C_CONTIGUOUS", "ALIGNED"]),
                            numpy.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim = 2, flags = ["C_CONTIGUOUS", "ALIGNED"]),
                            ctypes.c_size_t,
                            ctypes.c_size_t]
def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

def aligned(a, alignment=32):
    if (a.ctypes.data % alignment) == 0:
        return a
    extra = alignment // a.itemsize
    buf = numpy.empty(a.size + extra, dtype=a.dtype)
    ofs = (-buf.ctypes.data % alignment) // a.itemsize
    aa = buf[ofs:ofs+a.size].reshape(a.shape)
    numpy.copyto(aa, a)
    assert (aa.ctypes.data % alignment) == 0
    return aa

epsilon = 2**-50

if __name__ == '__main__':
    for arrSize in [8000]:
        arrA = numpy.array(numpy.random.rand(arrSize, arrSize),dtype=ctypes.c_double,order = 'C')
        arrB = numpy.array(numpy.random.rand(arrSize, arrSize),dtype=ctypes.c_double,order = 'C')
        #arrA = numpy.array(numpy.full((arrSize,arrSize),3.0),dtype=ctypes.c_double,order = 'C')
        #arrB = numpy.array(numpy.full((arrSize,arrSize),2.0),dtype=ctypes.c_double,order = 'C')
        arrResA = numpy.empty((arrSize, arrSize), dtype = ctypes.c_double, order = 'C')
        arrResB = numpy.empty((arrSize, arrSize), dtype = ctypes.c_double, order = 'C')
        arrResC = numpy.empty((arrSize, arrSize), dtype = ctypes.c_double, order = 'C')
        arrA = aligned(arrA)
        # No need to align arrB, since it will be transposed anyway
        wrapped = wrapper(matmullib.matmulOpenClNaive, arrA, arrB, arrResA, arrSize)
        t = timeit.timeit(wrapped, number=1)
        print("OpenCL", t)
        #wrapped = wrapper(matmullib.matmulSIMDMT, arrA, arrB, arrResB, arrSize, 6)
        #t = timeit.timeit(wrapped, number=1)
        #print("MT", t)
        #wrapped = wrapper(matmullib.simdMoreOptimized, arrA, arrB, arrResC, arrSize)
        #timeit.timeit(wrapped, number=1)
        #t = timeit.timeit(wrapped, number=1)
        #print("SIMD new", t)
