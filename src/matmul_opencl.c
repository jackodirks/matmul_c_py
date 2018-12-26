#include <stdio.h>
#include <stdlib.h>
#include <clext.h>
#include <stdbool.h>
#include <errno.h>
#include <string.h>
#include <math.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#define KERNELFILENAME "clKernel/matmulNaive.cl"
#define KERNELNAME "matmul"

static const char BUILDOPTIONS[] = "-Werror";

static bool initialized = false;

static cl_command_queue commandQueue;
static cl_context context;
static cl_kernel kernel;
static size_t workitem_size[3];
static size_t kernelWorkGroupSize;

static volatile bool contextError = false;
static volatile const char* contextErrInfo;

static void contextErrorCallback(const char* errInfo, const void* privateInfo, size_t cb, void* user_data){
    (void)user_data;
    (void)cb;
    (void)privateInfo;
    contextErrInfo = errInfo;
    contextError = true;
}

static void clError(const cl_int ret, const int lineNumber) {
    fprintf(stderr, "OpenCL error %d: %s at %s:%d\n", ret, clGetErrorString(ret), __FILE__, lineNumber);
    exit(1);
}

static void clCheckError(const cl_int ret, const int lineNumber){
    if (ret != CL_SUCCESS) return clError(ret, lineNumber);
}

static void checkContextError(void){
    if (!contextError) return;
    fprintf(stderr, "The context had an error: %s\n", contextErrInfo);
    exit(1);
}

void initialize(void){
    // Setup openCl
    cl_uint ret_num_platforms;
    cl_program program;
    cl_int ret;
    // First, get the number of platforms available
    ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);
    clCheckError(ret, __LINE__);
    if (ret_num_platforms == 0){
        fprintf(stderr, "No platforms available.\n");
        exit(1);
    }
    // Get the information of all platforms
    cl_platform_id platform_id[ret_num_platforms];
    ret = clGetPlatformIDs(ret_num_platforms, platform_id, NULL);
    clCheckError(ret, __LINE__);
    cl_device_id* device_id[ret_num_platforms];
    cl_uint ret_num_devices[ret_num_platforms];
    cl_uint deviceCount = 0;
    for (cl_uint i = 0; i < ret_num_platforms; ++i){
        ret = clGetDeviceIDs(platform_id[i], CL_DEVICE_TYPE_GPU, 0, NULL, &ret_num_devices[i]);
        if (ret != CL_DEVICE_NOT_FOUND && ret != CL_SUCCESS){
            clError(ret, __LINE__);
        } else if (ret == CL_DEVICE_NOT_FOUND){
            device_id[i] = NULL;
            continue;
        }
        deviceCount += ret_num_devices[i];
        device_id[i] = malloc(ret_num_devices[i]*sizeof(device_id));
        ret = clGetDeviceIDs(platform_id[i], CL_DEVICE_TYPE_GPU, ret_num_devices[i], device_id[i], NULL);
        clCheckError(ret, __LINE__);
    }
    if (deviceCount == 0){
        fprintf(stderr, "No GPU devices found\n");
        exit(1);
    }
    // Select the device. Heuristicly pick the device with the largest global memory.
    cl_device_id pickedDevice;
    cl_ulong maxGMem = 0;
    for (cl_uint i = 0; i < ret_num_platforms; ++i){
        for (cl_uint j = 0; j < ret_num_devices[i]; ++j){
            cl_ulong gMem;
            ret = clGetDeviceInfo(device_id[i][j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gMem), &gMem, NULL);
            clCheckError(ret, __LINE__);
            if (gMem > maxGMem){
                maxGMem = gMem;
                pickedDevice = device_id[i][j];
            }
        }
    }
    // Clean the allocated memory
    for (cl_uint i = 0; i < ret_num_platforms; ++i){
        if (device_id[i] != NULL){
            free(device_id[i]);
            device_id[i] = NULL;
        }
    }
    // Finally, create the context
    context = clCreateContext( NULL, 1, &pickedDevice, contextErrorCallback, NULL, &ret);
    clCheckError(ret, __LINE__);
    // Get some information from the device
    ret = clGetDeviceInfo(pickedDevice, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workitem_size), &workitem_size, NULL);
    clCheckError(ret, __LINE__);
    ret = clGetDeviceInfo(pickedDevice, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(kernelWorkGroupSize), &kernelWorkGroupSize, NULL);
    clCheckError(ret, __LINE__);
    commandQueue = clCreateCommandQueue(context, pickedDevice, 0, &ret);
    clCheckError(ret, __LINE__);

    // Read the OpenCL kernel from the file
    FILE *fp = fopen(KERNELFILENAME, "r");
    if (!fp) {
        fprintf(stderr, "%s:%d Failed to load kernel %s. Error: %d (%s)\n",__FILE__, __LINE__, KERNELFILENAME, errno, strerror(errno));
        exit(1);
    }
    if(fseek(fp, 0L, SEEK_END) == -1){
        fprintf(stderr, "%s:%d Failed to load kernel %s. Error: %d (%s)\n",__FILE__, __LINE__, KERNELFILENAME, errno, strerror(errno));
        exit(1);
    }
    size_t size = (size_t)ftell(fp);
    rewind(fp);
    char sourceStr[size + 1];
    sourceStr[size] = 0;
    size_t readBytes = fread(sourceStr, 1, size, fp);
    fclose(fp);
    // Create the OpenCL program
    const char *sourceStrPtr = sourceStr;
    program = clCreateProgramWithSource(context, 1, &sourceStrPtr, (const size_t *)&readBytes, &ret);
    clCheckError(ret, __LINE__);
    ret = clBuildProgram(program, 1, &pickedDevice, BUILDOPTIONS, NULL, NULL);
    if (ret != CL_SUCCESS){
        char errString[1000];
        fprintf(stderr, "Error while building\n");
        cl_int newRet = clGetProgramBuildInfo(program, pickedDevice, CL_PROGRAM_BUILD_LOG, sizeof(char)*1000, errString, NULL);
        if (newRet != CL_SUCCESS) clError(newRet, __LINE__);
        fprintf(stderr, "%s", errString);
        clError(ret, __LINE__);
    }
    kernel = clCreateKernel(program, KERNELNAME, &ret);
    clCheckError(ret, __LINE__);
    initialized = true;
}

void uninialize(void){
    initialized = false;
    cl_int ret;
    ret = clFlush(commandQueue);
    if (ret != CL_SUCCESS) clError(ret, __LINE__);
    ret = clFinish(commandQueue);
    if (ret != CL_SUCCESS) clError(ret, __LINE__);
    ret = clReleaseKernel(kernel);
    if (ret != CL_SUCCESS) clError(ret, __LINE__);
    ret = clReleaseCommandQueue(commandQueue);
    if (ret != CL_SUCCESS) clError(ret, __LINE__);
    ret = clReleaseContext(context);
    if (ret != CL_SUCCESS) clError(ret, __LINE__);
}

void matmulOpenClNaive(double* const restrict a, double* const restrict b, double* const restrict res, const size_t size){
    if (!initialized) initialize();
    double* c = malloc(size*size*sizeof(double));
    for (size_t i = 0; i < size; ++i){
        for (size_t j = 0; j < size; ++j){
            c[j + i*size] = b[i + j*size];
        }
    }
    // Create the required buffers
    cl_int ret;
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, size*size*sizeof(double), a, &ret);
    if (ret != CL_SUCCESS) clError(ret, __LINE__);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, size*size*sizeof(double), c, &ret);
    if (ret != CL_SUCCESS) clError(ret, __LINE__);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size * size * sizeof(double), NULL, &ret);
    if (ret != CL_SUCCESS) clError(ret, __LINE__);

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
    if (ret != CL_SUCCESS) clError(ret, __LINE__);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
    if (ret != CL_SUCCESS) clError(ret, __LINE__);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);
    if (ret != CL_SUCCESS) clError(ret, __LINE__);
    unsigned int mSize = size;
    ret = clSetKernelArg(kernel, 3, sizeof(mSize), &mSize);
    if (ret != CL_SUCCESS) clError(ret, __LINE__);
    size_t global_item_size[2] = {size, size};
    size_t local_item_size[2];
    local_item_size[0] = size*size > kernelWorkGroupSize ? sqrt(kernelWorkGroupSize) : size;
    local_item_size[0] = local_item_size[0] > workitem_size[0] ? workitem_size[0] : local_item_size[0];
    local_item_size[1] = size*size > kernelWorkGroupSize ? sqrt(kernelWorkGroupSize) : size;
    local_item_size[1] = local_item_size[1] > workitem_size[1] ? workitem_size[1] : local_item_size[1];
    if (global_item_size[0] % local_item_size[0] != 0) fprintf(stderr, "Warning: Amount of threads in dim 0 is not a multiple of %zu, so that the kernel will not run optimally\n", local_item_size[0]);
    while(global_item_size[0] % local_item_size[0] != 0) local_item_size[0]--;
    if (global_item_size[1] % local_item_size[1] != 0) fprintf(stderr, "Warning: Amount of threads in dim 1 is not a multiple of %zu, so that the kernel will not run optimally\n", local_item_size[1]);
    while(global_item_size[1] % local_item_size[1] != 0) local_item_size[1]--;
    cl_event waitEvents[2];
    ret = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, (const size_t*)global_item_size, (const size_t*)local_item_size, 0, NULL, &waitEvents[0]);
    checkContextError();
    if (ret != CL_SUCCESS) clError(ret, __LINE__);
    ret = clEnqueueReadBuffer(commandQueue, c_mem_obj, CL_TRUE, 0, size*size * sizeof(double), res, 1, &waitEvents[0], &waitEvents[1]);
    if (ret != CL_SUCCESS) clError(ret, __LINE__);
    ret = clWaitForEvents(2, waitEvents);
    if (ret != CL_SUCCESS) clError(ret, __LINE__);
    ret = clReleaseMemObject(a_mem_obj);
    if (ret != CL_SUCCESS) clError(ret, __LINE__);
    ret = clReleaseMemObject(b_mem_obj);
    if (ret != CL_SUCCESS) clError(ret, __LINE__);
    ret = clReleaseMemObject(c_mem_obj);
    if (ret != CL_SUCCESS) clError(ret, __LINE__);
    free(c);
}
