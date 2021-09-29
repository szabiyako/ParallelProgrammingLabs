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
        printf("����� ����������: %d\n", device);
        printf("��� ����������: %s\n", deviceProp.name);
        printf("����� ���������� ������: %d �����\n",
            deviceProp.totalGlobalMem / 1024 / 1024);
        printf("����� shared-������ � ����� : %d ���� (%d �����)\n",
            deviceProp.sharedMemPerBlock, deviceProp.sharedMemPerBlock / 1024);
        printf("����� ����������� ������: %d\n", deviceProp.regsPerBlock);
        printf("������ warp'a: %d\n", deviceProp.warpSize);
        printf("������ ���� ������: %d (%d �����)\n",
            deviceProp.memPitch, deviceProp.memPitch / 1024 / 1024);
        printf("���� ���������� ������� � �����: %d\n",
            deviceProp.maxThreadsPerBlock);
        printf("������������ ����������� ������: x = %d, y = %d, z = %d\n",
            deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("������������ ������ �����: x = %d, y = %d, z = %d\n",
            deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("�������� �������: %d ��� (%d ���)\n",
            deviceProp.clockRate, deviceProp.clockRate / 1000);
        printf("����� ����� ����������� ������: %d\n",
            deviceProp.totalConstMem);
        printf("�������������� ��������: %d.%d\n", deviceProp.major,
            deviceProp.minor);
        printf("�������� ����������� ������������ : %d\n",
            deviceProp.textureAlignment);
        printf("���������� �����������: %d\n",
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
    while ((N <= 0) || (N > 46340)) {
        while (std::cout << "Enter N = " && !(std::cin >> N)) {
            std::cin.clear(); //clear bad input flag
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); //discard input
            std::cout << "Invalid input\n";
        }
        if (N <= 0)
            std::cout << "Invalid input\n";
        else if (N > 46340)
            std::cout << "N = " << N << " is greater than maxN = 46340\n";
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
    function(result, arr, size); // First start is taking much longer than others, so just skip it

    clearArray(result, size);
    const std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();
    function(result, arr, size);
    const std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
    printf("----------------------------\n");
    printf(" %s:\n", name);
    printf("----------------------------\n");
    clearArray(result, size);
    testFunction(result, arr, size); // Prints all steps timings
    if (usePrint)
        printArray(result, size);
    printf("Elapsed time: %d s\n              %d ms\n              %d ns\n",
        std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count(),
        std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count());
    fflush(stdout);
}
