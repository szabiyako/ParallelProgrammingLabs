#include "gpu.cuh"

#include <algorithm>

#include "../Cpu/cpu.h"


__global__ static void quicksort(int64_t *const values, const int64_t size) {
#define MAX_LEVELS	300

	int64_t pivot, L, R;
	int64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	int64_t start[MAX_LEVELS];
	int64_t end[MAX_LEVELS];

	start[idx] = idx;
	end[idx] = size - 1;
	while (idx >= 0) {
		L = start[idx];
		R = end[idx];
		if (L < R) {
			pivot = values[L];
			while (L < R) {
				while (values[R] >= pivot && L < R)
					R--;
				if (L < R)
					values[L++] = values[R];
				while (values[L] < pivot && L < R)
					L++;
				if (L < R)
					values[R--] = values[L];
			}
			values[L] = pivot;
			start[idx + 1] = L + 1;
			end[idx + 1] = end[idx];
			end[idx++] = L;
			if (end[idx] - start[idx] > end[idx - 1] - start[idx - 1]) {
				// swap start[idx] and start[idx-1]
				int64_t tmp = start[idx];
				start[idx] = start[idx - 1];
				start[idx - 1] = tmp;

				// swap end[idx] and end[idx-1]
				tmp = end[idx];
				end[idx] = end[idx - 1];
				end[idx - 1] = tmp;
			}

		}
		else
			idx--;
	}
}


void Sort::cudaSingle(int64_t *const arr, const int64_t size)
{
	int64_t *cudaArr;
	cudaMalloc((void **)&cudaArr, size * sizeof(int64_t));
	cudaMemcpy(cudaArr, arr, size * sizeof(int64_t), cudaMemcpyHostToDevice);
	quicksort <<<1, 1, 1>>> (cudaArr, size);
	cudaThreadSynchronize();
	cudaMemcpy(arr, cudaArr, size * sizeof(int64_t), cudaMemcpyDeviceToHost);
	cudaFree(cudaArr);
}
