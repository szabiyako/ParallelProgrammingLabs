#include "solveCpu.h"

void Solve::cpu(int* res, const int* matrix, const int sideSize)
{
    for (int col = 0; col < sideSize; ++col) {
        for (int row = 0; row < sideSize; ++row) {
            const int elemIdx = col * sideSize + row;
            if (matrix[elemIdx] == 0)
                ++res[col];
        }
    }
}
