#ifndef SOLVE_CPU_OPENMP_H
#define SOLVE_CPU_OPENMP_H

#include <stdint.h>

namespace Solve {

void cpuOpenMpStatic(int64_t* res, const char* matrix, const int64_t sideSize);
void cpuOpenMpDynamic(int64_t* res, const char* matrix, const int64_t sideSize);
void cpuOpenMpGuided(int64_t* res, const char* matrix, const int64_t sideSize);

}

#endif // SOLVE_CPU_OPENMP_H