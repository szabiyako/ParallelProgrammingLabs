#ifndef SOVE_CUDA_CONSTANT_MEMORY_H
#define SOVE_CUDA_CONSTANT_MEMORY_H

#include "../../../cudaInclude.h"

namespace Solve {

void cudaConstantMemory(int *res, const int *matrix, const int sideSize);
void testCudaConstantMemory(int* res, const int* matrix, const int sideSize);

namespace Internal {

__global__ void computeConstant(int *res, const int size);

cudaError_t cudaConstantMemory(int* res, const int* arr, const int size);

// Same function as cuda(...) but with timers for each step and printf for each timer
cudaError_t testCudaConstantMemory(int* res, const int* arr, const int size);

}

}

#endif // SOVE_CUDA_CONSTANT_MEMORY_H