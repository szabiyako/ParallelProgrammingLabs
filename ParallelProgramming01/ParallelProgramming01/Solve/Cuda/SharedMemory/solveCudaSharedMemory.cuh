#ifndef SOVE_CUDA_SHARED_MEMORY_H
#define SOVE_CUDA_SHARED_MEMORY_H

#include "../../../cudaInclude.h"

namespace Solve {

void cudaSharedMemory(int *res, const int *matrix, const int sideSize);
void testCudaSharedMemory(int* res, const int* matrix, const int sideSize);

namespace Internal {

__global__ void computeShared(int *res, const int *arr, const int size);

cudaError_t cudaSharedMemory(int* res, const int* arr, const int size);

// Same function as cuda(...) but with timers for each step and printf for each timer
cudaError_t testCudaSharedMemory(int* res, const int* arr, const int size);

}

}

#endif // SOVE_CUDA_SHARED_MEMORY_H