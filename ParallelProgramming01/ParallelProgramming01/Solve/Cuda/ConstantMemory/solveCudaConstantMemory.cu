#include "solveCudaConstantMemory.cuh"

#include <stdio.h>
#include <chrono>

const int maxSideSize = 128;
const int maxElements = maxSideSize * maxSideSize;

__constant__ int dev_arr[maxElements];

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                 SOLVER                                                           //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Solve::cudaConstantMemory(int *res, const int *matrix, const int sideSize)
{
	cudaError_t cudaStatus = Solve::Internal::cudaConstantMemory(res, matrix, sideSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Solve::cuda failed!\n");
		fflush(stdout);
		return;
	}
}

void Solve::testCudaConstantMemory(int* res, const int* matrix, const int sideSize)
{
    cudaError_t cudaStatus = Solve::Internal::testCudaConstantMemory(res, matrix, sideSize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Solve::cuda failed!\n");
        fflush(stdout);
        return;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                               INTERNAL                                                           //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void Solve::Internal::computeConstant(int* res, const int size)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < size) {
        for (int row = 0; row < size; ++row) {
            const int elemIdx = col * size + row;
            if (dev_arr[elemIdx] == 0)
                ++res[col];
        }
    }
}


///////////////////////////////
//       BASE FUNCTION       //
///////////////////////////////
cudaError_t Solve::Internal::cudaConstantMemory(int* res, const int* arr, const int size)
{
    int* dev_res = 0;
    cudaError_t cudaStatus;

    if (size >= maxSideSize) {
        //printf("Constant memory overflow, max size is %d, but received %d\n", maxSideSize - 1, size);
        //fflush(stdout);
        return cudaError::cudaSuccess;
    }

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
        goto Error;
    }

    
    // Allocate GPU buffers for 2 arrays (one input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_res, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        goto Error;
    }
   
    
    // Copy input array from host memory to GPU buffers.
    cudaStatus = cudaMemcpyToSymbol(dev_arr, arr, size * size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol failed!\n");
        goto Error;
    }
    

    const int block_size = deviceProp.maxThreadsPerBlock;
    const int num_blocks = size / block_size + 1;
    int resBlockSize = block_size;
    if (num_blocks == 1)
        resBlockSize = size;

    
    // Launch a kernel on the GPU with one thread for each column.
    computeConstant <<<num_blocks, resBlockSize >>> (dev_res, resBlockSize);
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    
    // Copy output array from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(res, dev_res, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        goto Error;
    }

Error:
    cudaFree(dev_res);

    return cudaStatus;
}

///////////////////////////////
//       TEST FUNCTION       //
///////////////////////////////
cudaError_t Solve::Internal::testCudaConstantMemory(int* res, const int* arr, const int size)
{
    int* dev_res = 0;
    cudaError_t cudaStatus;

    if (size >= maxSideSize) {
        printf("Constant memory overflow, max size is %d, but received %d\n", maxSideSize - 1, size);
        fflush(stdout);
        return cudaError::cudaErrorAssert;
    }

    cudaEvent_t eAllocStart, eAllocStop;
    cudaEvent_t eCopyStart, eCopyStop;
    cudaEvent_t eComputeStart, eComputeStop;
    cudaEvent_t eReciveStart, eReciveStop;
    cudaEvent_t eFreeStart, eFreeStop;

    cudaEventCreate(&eAllocStart);
    cudaEventCreate(&eAllocStop);

    cudaEventCreate(&eCopyStart);
    cudaEventCreate(&eCopyStop);

    cudaEventCreate(&eComputeStart);
    cudaEventCreate(&eComputeStop);

    cudaEventCreate(&eReciveStart);
    cudaEventCreate(&eReciveStop);

    cudaEventCreate(&eFreeStart);
    cudaEventCreate(&eFreeStop);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
        goto Error;
    }


    const std::chrono::system_clock::time_point startTimeAlloc = std::chrono::system_clock::now();
    // Allocate GPU buffers for two arrays (one input, one output)    .
    cudaEventRecord(eAllocStart);
    cudaStatus = cudaMalloc((void**)&dev_res, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        goto Error;
    }
    const std::chrono::system_clock::time_point endTimeAlloc = std::chrono::system_clock::now();


    const std::chrono::system_clock::time_point startTimeCopy = std::chrono::system_clock::now();
    // Copy input arrays from host memory to GPU buffers.
    cudaEventRecord(eCopyStart);
    cudaStatus = cudaMemcpyToSymbol(dev_arr, arr, size * size * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(eCopyStop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol failed!\n");
        goto Error;
    }
    const std::chrono::system_clock::time_point endTimeCopy = std::chrono::system_clock::now();


    const int block_size = deviceProp.maxThreadsPerBlock;
    const int num_blocks = size / block_size + 1;
    int resBlockSize = block_size;
    if (num_blocks == 1)
        resBlockSize = size;


    const std::chrono::system_clock::time_point startTimeCompute = std::chrono::system_clock::now();
    // Launch a kernel on the GPU with one thread for each column.
    cudaEventRecord(eComputeStart);
    computeConstant <<<num_blocks, resBlockSize >> > (dev_res, resBlockSize);
    cudaEventRecord(eComputeStop);
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
    const std::chrono::system_clock::time_point endTimeCompute = std::chrono::system_clock::now();


    const std::chrono::system_clock::time_point startTimeRecive = std::chrono::system_clock::now();
    // Copy output array from GPU buffer to host memory.
    cudaEventRecord(eReciveStart);
    cudaStatus = cudaMemcpy(res, dev_res, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(eReciveStop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        goto Error;
    }
    const std::chrono::system_clock::time_point endTimeRecive = std::chrono::system_clock::now();


Error:
    const std::chrono::system_clock::time_point startTimeFree = std::chrono::system_clock::now();
    cudaEventRecord(eFreeStart);
    cudaFree(dev_res);
    cudaEventRecord(eFreeStop);
    const std::chrono::system_clock::time_point endTimeFree = std::chrono::system_clock::now();


    float eAllocTime;
    float eCopyTime;
    float eComputeTime;
    float eReciveTime;
    float eFreeTime;

    cudaEventSynchronize(eAllocStop);
    cudaEventSynchronize(eCopyStop);
    cudaEventSynchronize(eComputeStop);
    cudaEventSynchronize(eReciveStop);
    cudaEventSynchronize(eFreeStop);
    cudaEventElapsedTime(&eAllocTime, eAllocStart, eAllocStop);
    cudaEventElapsedTime(&eCopyTime, eCopyStart, eCopyStop);
    cudaEventElapsedTime(&eComputeTime, eComputeStart, eComputeStop);
    cudaEventElapsedTime(&eReciveTime, eReciveStart, eReciveStop);
    cudaEventElapsedTime(&eFreeTime, eFreeStart, eFreeStop);

    printf("Alloc time CUDA:   %d s\n                   %d ms\n                   %f ms (CUDA events)\n                   %d ns\n",
        std::chrono::duration_cast<std::chrono::seconds>(endTimeAlloc - startTimeAlloc).count(),
        std::chrono::duration_cast<std::chrono::milliseconds>(endTimeAlloc - startTimeAlloc).count(),
        eAllocTime,
        std::chrono::duration_cast<std::chrono::nanoseconds>(endTimeAlloc - startTimeAlloc).count());
    printf("Copy time CUDA:    %d s\n                   %d ms\n                   %f ms (CUDA events)\n                   %d ns\n",
        std::chrono::duration_cast<std::chrono::seconds>(endTimeCopy - startTimeCopy).count(),
        std::chrono::duration_cast<std::chrono::milliseconds>(endTimeCopy - startTimeCopy).count(),
        eCopyTime,
        std::chrono::duration_cast<std::chrono::nanoseconds>(endTimeCopy - startTimeCopy).count());
    printf("Compute time CUDA: %d s\n                   %d ms\n                   %f ms (CUDA events)\n                   %d ns\n",
        std::chrono::duration_cast<std::chrono::seconds>(endTimeCompute - startTimeCompute).count(),
        std::chrono::duration_cast<std::chrono::milliseconds>(endTimeCompute - startTimeCompute).count(),
        eComputeTime,
        std::chrono::duration_cast<std::chrono::nanoseconds>(endTimeCompute - startTimeCompute).count());
    printf("Recive time CUDA:  %d s\n                   %d ms\n                   %f ms (CUDA events)\n                   %d ns\n",
        std::chrono::duration_cast<std::chrono::seconds>(endTimeRecive - startTimeRecive).count(),
        std::chrono::duration_cast<std::chrono::milliseconds>(endTimeRecive - startTimeRecive).count(),
        eReciveTime,
        std::chrono::duration_cast<std::chrono::nanoseconds>(endTimeRecive - startTimeRecive).count());
    printf("Free time CUDA:    %d s\n                   %d ms\n                   %f ms (CUDA events)\n                   %d ns\n",
        std::chrono::duration_cast<std::chrono::seconds>(endTimeFree - startTimeFree).count(),
        std::chrono::duration_cast<std::chrono::milliseconds>(endTimeFree - startTimeFree).count(),
        eFreeTime,
        std::chrono::duration_cast<std::chrono::nanoseconds>(endTimeFree - startTimeFree).count());
    fflush(stdout);
    return cudaStatus;
}
