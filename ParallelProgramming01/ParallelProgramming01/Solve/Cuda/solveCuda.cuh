#ifndef SOVE_CUDA_H
#define SOVE_CUDA_H

#include "../../cudaInclude.h"

namespace Solve {

void cuda(int *res, const int *matrix, const int sideSize);
void testCuda(int* res, const int* matrix, const int sideSize);

namespace Internal {

__global__ void compute(int *res, const int *arr, const int size);

cudaError_t cuda(int* res, const int* arr, const int size);
cudaError_t testCuda(int* res, const int* arr, const int size);

}

}

#endif // SOVE_CUDA_H