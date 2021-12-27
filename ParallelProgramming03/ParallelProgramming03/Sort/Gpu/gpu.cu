#include "gpu.cuh"

#define THREADS_PER_BLOCK 1024

typedef struct vars {
    int64_t l;
    int64_t r;
    int64_t leq;
} vars;

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
////////
__global__ void gpuPartitionSwap(int64_t *input, int64_t *output, vars *endpts,
    int64_t pivot, int64_t l, int64_t r,
    int64_t d_leq[],
    int64_t d_gt[], int64_t *d_leq_val, int64_t *d_gt_val,
    int64_t nBlocks)
{
    //copy a section of the input into shared memory
    __shared__ int64_t bInput[THREADS_PER_BLOCK];
    __syncthreads();
    int64_t idx = l + blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    __shared__ int64_t lThisBlock, rThisBlock;
    __shared__ int64_t lOffset, rOffset;

    if (threadIdx.x == 0) {
        d_leq[blockIdx.x] = 0;
        d_gt[blockIdx.x] = 0;
        *d_leq_val = 0;
        *d_gt_val = 0;
    }
    __syncthreads();

    if (idx <= (r - 1)) {
        bInput[threadIdx.x] = input[idx];

        //make comparison against the pivot, setting 'status' and updating the counter (if necessary)
        if (bInput[threadIdx.x] <= pivot) {
            //atomicAdd(&(d_leq[blockIdx.x]), 1);
            atomicAdd((uint64_t*)&(d_leq[blockIdx.x]), 1);
        }
        else {
            //atomicAdd(&(d_gt[blockIdx.x]), 1);
            atomicAdd((uint64_t *)&(d_gt[blockIdx.x]), 1);
        }

    }
    __syncthreads();


    if (threadIdx.x == 0) {
        lThisBlock = d_leq[blockIdx.x];
        lOffset = l + atomicAdd((uint64_t *)d_leq_val, lThisBlock);
    }
    if (threadIdx.x == 1) {
        rThisBlock = d_gt[blockIdx.x];
        rOffset = r - atomicAdd((uint64_t *)d_gt_val, rThisBlock);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        int64_t m = 0;
        int64_t n = 0;
        for (int64_t j = 0; j < THREADS_PER_BLOCK; j++) {
            int64_t chk = l + blockIdx.x * THREADS_PER_BLOCK + j;
            if (chk <= (r - 1)) {
                if (bInput[j] <= pivot) {
                    //     bInput[j], lOffset+m, lOffset, m);
                    output[lOffset + m] = bInput[j];
                    ++m;
                }
                else {
                    //      bInput[j], rOffset-n, rOffset, n);
                    output[rOffset - n] = bInput[j];
                    ++n;
                }
            }
        }
    }

    __syncthreads();

    if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
        int64_t pOffset = l;
        for (int64_t k = 0; k < nBlocks; k++)
            pOffset += d_leq[k];

        output[pOffset] = pivot;
        endpts->l = (pOffset - 1);
        endpts->r = (pOffset + 1);
    }

    return;
}

void gqSort(int64_t ls[], int64_t l, int64_t r, int64_t length)
{
    if ((r - l) >= 1) {
        int64_t pivot = ls[r];

        int64_t numBlocks = (r - l) / THREADS_PER_BLOCK;
        if ((numBlocks * THREADS_PER_BLOCK) < (r - l))
            numBlocks++;

        int64_t *d_ls;
        int64_t *d_ls2;
        vars endpts;
        endpts.l = l;
        endpts.r = r;

        vars *d_endpts;
        int64_t *d_leq, *d_gt, *d_leq_val, *d_gt_val;
        int64_t size = sizeof(int64_t);
        cudaMalloc(&(d_ls), size * length);
        cudaMalloc(&(d_ls2), size * length);
        cudaMalloc(&(d_endpts), sizeof(vars));
        cudaMalloc(&(d_leq), 4 * numBlocks);
        cudaMalloc(&(d_gt), 4 * numBlocks);
        cudaMalloc(&d_leq_val, 4);
        cudaMalloc(&d_gt_val, 4);
        cudaMemcpy(d_ls, ls, size * length, cudaMemcpyHostToDevice);
        cudaMemcpy(d_ls2, ls, size * length, cudaMemcpyHostToDevice);

        gpuPartitionSwap <<<numBlocks, THREADS_PER_BLOCK >>> (d_ls, d_ls2, d_endpts, pivot, l, r, d_leq, d_gt, d_leq_val, d_gt_val, numBlocks);

        cudaMemcpy(ls, d_ls2, size * length, cudaMemcpyDeviceToHost);
        cudaMemcpy(&(endpts), d_endpts, sizeof(vars), cudaMemcpyDeviceToHost);

        cudaThreadSynchronize();

        cudaFree(d_ls);
        cudaFree(d_ls2);
        cudaFree(d_endpts);
        cudaFree(d_leq);
        cudaFree(d_gt);

        if (endpts.l >= l)
            gqSort(ls, l, endpts.l, length);
        if (endpts.r <= r)
            gqSort(ls, endpts.r, r, length);

    }

    return;
}
////////

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

void Sort::cuda(int64_t *const arr, const int64_t size)
{
    gqSort(arr, 0, size - 1, size);
}