#include "tools.h"

#include <stdio.h>
#include <iostream>
#include <ctime>
#include <chrono>

#include "../cudaInclude.h"

using namespace Tools;

void Tools::printCudaDevicesInfo()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int device = 0; device < deviceCount; device++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Номер устройства: %d\n", device);
        printf("Имя устройства: %s\n", deviceProp.name);
        printf("Объем глобальной памяти: %d Мбайт\n",
            deviceProp.totalGlobalMem / 1024 / 1024);
        printf("Объем shared-памяти в блоке : %d Байт (%d Кбайт)\n",
            deviceProp.sharedMemPerBlock, deviceProp.sharedMemPerBlock / 1024);
        printf("Объем регистровой памяти: %d\n", deviceProp.regsPerBlock);
        printf("Размер warp'a: %d\n", deviceProp.warpSize);
        printf("Размер шага памяти: %d (%d Гбайт)\n",
            deviceProp.memPitch, deviceProp.memPitch / 1024 / 1024);
        printf("Макс количество потоков в блоке: %d\n",
            deviceProp.maxThreadsPerBlock);
        printf("Максимальная размерность потока: x = %d, y = %d, z = %d\n",
            deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("Максимальный размер сетки: x = %d, y = %d, z = %d\n",
            deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("Тактовая частота: %d кГц (%d мГц)\n",
            deviceProp.clockRate, deviceProp.clockRate / 1000);
        printf("Общий объем константной памяти: %d\n",
            deviceProp.totalConstMem);
        printf("Вычислительная мощность: %d.%d\n", deviceProp.major,
            deviceProp.minor);
        printf("Величина текстурного выравнивания : %d\n",
            deviceProp.textureAlignment);
        printf("Количество процессоров: %d\n",
            deviceProp.multiProcessorCount);
    }
    fflush(stdout);
}

void Tools::printMatrix(int *arr, const int sideSize)
{
    for (int row = 0; row < sideSize; ++row) {
        std::cout << "|";
        for (int col = 0; col < sideSize; ++col) {
            const int elemIdx = col * sideSize + row;
            std::cout.width(4);
            std::cout << arr[elemIdx];
            std::cout << " ";
        }
        std::cout << "\b|" << std::endl;
    }
}

void Tools::printArray(int *arr, const int size)
{
    const int last = size - 1;
    printf("[");
    for (int i = 0; i < last; ++i) {
        printf("%d, ", arr[i]);
    }
    printf("%d]\n", arr[last]);
    fflush(stdout);
}

int Tools::getSizeFromInput()
{
    int N = -1;
    while (N <= 0) {
        while (std::cout << "Enter N = " && !(std::cin >> N)) {
            std::cin.clear(); //clear bad input flag
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); //discard input
            std::cout << "Invalid input\n";
        }
        if (N <= 0)
            std::cout << "Invalid input\n";
    }
    return N;
}

void Tools::fillArrayRandom(int *arr, const int size, const int minValue, const int maxValue)
{
    for (int i = 0; i < size; ++i) {
        arr[i] = getRandomInt(minValue, maxValue);
    }
}

void Tools::clearArray(int *arr, const int size)
{
    for (int i = 0; i < size; ++i) {
        arr[i] = 0;
    }
}

void Tools::setupRandomizer()
{
    srand(time(0));
}

int Tools::getRandomInt(const int min, const int max)
{
    const int range = std::abs(max - min) + 1;
    return (rand() % range) + min;
}

void Tools::testFunction(
    void (*function)(int*, const int*, const int),
    const char* name,
    int* result,
    const int* arr,
    const int size,
    const bool usePrint)
{
    //Clear result
    clearArray(result, size);

    const std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();
    function(result, arr, size);
    const std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
    printf("--------------\n");
    printf(" %s:\n", name);
    printf("--------------\n");
    if (usePrint)
        printArray(result, size);
    printf("Elapsed time: %d s\n              %d ms\n              %d ns\n",
        std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count(),
        std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count());
    fflush(stdout);
}

void Tools::testFunctionCuda(
    void (*function)(int*, const int*, const int),
    void (*testFunction)(int*, const int*, const int),
    const char* name,
    int* result,
    const int* arr,
    const int size,
    const bool usePrint)
{
    //Clear result
    clearArray(result, size);
    function(result, arr, size);
    clearArray(result, size);
    const std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();
    function(result, arr, size);
    const std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
    printf("--------------\n");
    printf(" %s:\n", name);
    printf("--------------\n");
    clearArray(result, size);
    testFunction(result, arr, size);
    if (usePrint)
        printArray(result, size);
    printf("Elapsed time: %d s\n              %d ms\n              %d ns\n",
        std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count(),
        std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count());
    fflush(stdout);
}
