#ifndef TOOLS_H
#define TOOLS_H

#include <string>
#include <stdint.h>

namespace Tools {

void printCpuInfo();

void printArray(int64_t *arr, const int64_t size);

int64_t getSizeFromInput();
int64_t getIterationsFromInput();
void fillArrayRandom(int64_t *arr, const int64_t size, const int64_t minValue, const int64_t maxValue);
void clearArray(int64_t *arr, const int64_t size);

void setupRandomizer();
int64_t getRandomInt(const int64_t min, const int64_t max);

void testFunctionVerbose(
    void (*function)(int64_t *const, const int64_t),
    const char* name,
    int64_t *const result,
    int64_t *const initArr,
    const int64_t size,
    const bool usePrint,
    const int64_t iterations);

void testFunction(
    void (*function)(int64_t *const, const int64_t),
    int64_t *const result,
    int64_t *const initArr,
    const int64_t size,
    const int64_t iterations,
    size_t& avgTime,
    size_t& maxTime,
    size_t& minTime);

std::string getConvertedTime(const size_t timeInNs = 0);
std::string getMemoryAsString(const size_t bytes);

//void cmdMode(int argc, char* argv[]);

}

#endif // TOOLS_H
