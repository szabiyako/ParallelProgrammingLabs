#ifndef SOVE_CUDA_GLOBAL_MEMORY_H
#define SOVE_CUDA_GLOBAL_MEMORY_H

#include "../../../cudaInclude.h"

namespace Solve {

void cudaGlobalMemory(int *res, const int *matrix, const int sideSize);
void testCudaGlobalMemory(int* res, const int* matrix, const int sideSize);

namespace Internal {

__global__ void computeGlobal(int *res, const int *arr, const int size);

cudaError_t cudaGlobalMemory(int* res, const int* arr, const int size);

// Same function as cuda(...) but with timers for each step and printf for each timer
cudaError_t testCudaGlobalMemory(int* res, const int* arr, const int size);

}

}

#endif // SOVE_CUDA_GLOBAL_MEMORY_H