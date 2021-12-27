#include "solveCpuOpenMP.h"

#include "omp.h"

void Solve::cpuOpenMpStatic(int64_t* res, const char* matrix, const int64_t sideSize)
{
    #pragma omp parallel for schedule(static)
    for (int64_t col = 0; col < sideSize; ++col) {
        for (int64_t row = 0; row < sideSize; ++row) {
            const int64_t elemIdx = col * sideSize + row;
            if (matrix[elemIdx] == 0)
                ++res[col];
        }
    }
}

void Solve::cpuOpenMpDynamic(int64_t* res, const char* matrix, const int64_t sideSize)
{
    #pragma omp parallel for schedule(dynamic)
    for (int64_t col = 0; col < sideSize; ++col) {
        for (int64_t row = 0; row < sideSize; ++row) {
            const int64_t elemIdx = col * sideSize + row;
            if (matrix[elemIdx] == 0)
                ++res[col];
        }
    }
}

void Solve::cpuOpenMpGuided(int64_t* res, const char* matrix, const int64_t sideSize)
{
    #pragma omp parallel for schedule(guided)
    for (int64_t col = 0; col < sideSize; ++col) {
        for (int64_t row = 0; row < sideSize; ++row) {
            const int64_t elemIdx = col * sideSize + row;
            if (matrix[elemIdx] == 0)
                ++res[col];
        }
    }
}
