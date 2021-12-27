#include "solveCpu.h"

void Solve::cpu(int64_t* res, const char* matrix, const int64_t sideSize)
{
    for (int64_t col = 0; col < sideSize; ++col) {
        for (int64_t row = 0; row < sideSize; ++row) {
            const int64_t elemIdx = col * sideSize + row;
            if (matrix[elemIdx] == 0)
                ++res[col];
        }
    }
}
