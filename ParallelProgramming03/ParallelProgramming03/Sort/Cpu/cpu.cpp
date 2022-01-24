#include "cpu.h"

#include <algorithm>
#include <queue>

#include <omp.h>


static int partition(int64_t *const arr, int64_t low, int64_t high)
{
    const int64_t pivot = arr[high];    // pivot
    int64_t leftIdx = (low - 1);  // Index of smaller element

    for (int64_t rightIdx = low; rightIdx <= high - 1; rightIdx++)
    {
        if (arr[rightIdx] <= pivot)
        {
            leftIdx++;    // increment index of smaller element
            std::swap(arr[leftIdx], arr[rightIdx]);
        }
    }
    std::swap(arr[leftIdx + 1], arr[high]);
    return (leftIdx + 1);
}


static void quickSort(int64_t *const arr, int64_t low, int64_t high)
{
    if (low < high)
    {
        int64_t pi = partition(arr, low, high);

        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

void Sort::singleThread(int64_t *const arr, const int64_t size)
{
    quickSort(arr, 0, size - 1);
}







// PARALLEL

static void quickSortParallel(int64_t *const arr, const int64_t low, const int64_t high, const int64_t nTasks)
{
    if (low < high)
    {
        int64_t pi = partition(arr, low, high);

        #pragma omp task shared(arr) if(high - low > nTasks) 
            quickSort(arr, low, pi - 1);
        #pragma omp task shared(arr) if(high - low > nTasks) 
            quickSort(arr, pi + 1, high);
    }
}

void Sort::multiThread(int64_t *const arr, const int64_t size)
{
    const int64_t nTasks = size / 16;
    omp_set_dynamic(0);
    //omp_set_num_threads(16);
    #pragma omp parallel
    {
        #pragma omp single
        quickSortParallel(arr, 0, size - 1, nTasks);
    }
}