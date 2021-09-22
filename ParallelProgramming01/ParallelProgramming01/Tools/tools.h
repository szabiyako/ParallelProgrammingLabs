#ifndef TOOLS_H
#define TOOLS_H

namespace Tools {

void printCudaDevicesInfo();
void printMatrix(int *arr, const int size);
void printArray(int *arr, const int size);

int getSizeFromInput();
void fillArrayRandom(int *arr, const int size, const int minValue, const int maxValue);
void clearArray(int *arr, const int size);

void setupRandomizer();
int getRandomInt(const int min, const int max);

void testFunction(
    void (*function)(int*, const int*, const int),
    const char* name,
    int* result,
    const int* arr,
    const int size,
    const bool usePrint);

void testFunctionCuda(
    void (*function)(int*, const int*, const int),
    void (*testFunction)(int*, const int*, const int),
    const char* name,
    int* result,
    const int* arr,
    const int size,
    const bool usePrint);

}

#endif // TOOLS_H
