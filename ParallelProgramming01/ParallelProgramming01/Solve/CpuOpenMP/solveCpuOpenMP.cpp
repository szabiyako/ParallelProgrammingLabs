#include "solveCpuOpenMP.h"

#include "omp.h"
//#include <iostream>

void Solve::cpuOpenMP(int* res, const int* matrix, const int sideSize)
{
#pragma omp parallel for num_threads(32)
    for (int col = 0; col < sideSize; ++col) {
        for (int row = 0; row < sideSize; ++row) {
            const int elemIdx = col * sideSize + row;
            if (matrix[elemIdx] == 0)
                ++res[col];
            //std::cout << "index[" << col << "][" << row << "]" << std::endl;
        }
    }
}
