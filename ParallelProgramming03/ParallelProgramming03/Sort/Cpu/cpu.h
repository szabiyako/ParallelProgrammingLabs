#ifndef SORT_CPU_H
#define SORT_CPU_H

#include <stdint.h>

namespace Sort {

void singleThread(int64_t *const arr, const int64_t size);
void multiThread(int64_t *const arr, const int64_t size);

}

#endif // SOLVE_CPU_H