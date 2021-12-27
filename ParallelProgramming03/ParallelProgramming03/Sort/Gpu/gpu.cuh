#ifndef SORT_GPU_CUH
#define SORT_GPU_CUH

#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include <stdint.h>

namespace Sort {

void cudaSingle(int64_t *const arr, const int64_t size);
void cuda(int64_t *const arr, const int64_t size);

}

#endif // SORT_GPU_CUH