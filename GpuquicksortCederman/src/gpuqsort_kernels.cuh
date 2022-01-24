#include "gpuqsort.h"
#include "cuda_runtime_api.h"

__device__ inline void swap(unsigned int &a, unsigned int &b);

__device__ inline
void bitonicSort(unsigned int *fromvalues, unsigned int *tovalues, unsigned int from, unsigned int size);

__device__ inline void cumcount(unsigned int *lblock, unsigned int *rblock);

__global__ void part1(unsigned int *data, Params<unsigned int> *params, struct Hist *hist, Length<unsigned int> *lengths);

__global__ void part2(unsigned int *data, unsigned int *data2, struct Params<unsigned int> *params, struct Hist *hist, Length<unsigned int> *lengths);

__global__ void part3(unsigned int *data, struct Params<unsigned int> *params, struct Hist *hist, Length<unsigned int> *lengths);

__global__ void lqsort(unsigned int *adata, unsigned int *adata2, struct LQSortParams *bs, unsigned int phase);