/*
* Вариант № 11. Написать программу с использованием технологии CUDA, которая реализует следующие действия:
* формирует массив,
* размерностью N x N и формирует новый массив, элементы которого равны количеству нулевых элементов в соответствующем столбце исходного массива.
*/
#define NOMINMAX

#include <stdio.h>

#include <windows.h>
#include <conio.h>
#include <chrono>
#include <iostream>
#include <limits>

#include "Tools/tools.h"
#include "Solve/Cuda/solveCuda.cuh"
#include "Solve/Cpu/solveCpu.h"
#include "Solve/CpuOpenMP/solveCpuOpenMP.h"

int main()
{
    SetConsoleCP(1251);
    SetConsoleOutputCP(1251);

    //cudaSetDevice(0);

    Tools::setupRandomizer();
    printf("----------\n");
    printf("GPUs info:\n");
    printf("----------\n");
    fflush(stdout);
    Tools::printCudaDevicesInfo();
    printf("\n\n\n");
    printf("Вариант № 11. Написать программу с использованием технологии CUDA, которая реализует следующие действия:\n");
    printf("формирует массив, размерностью N x N и формирует новый массив,\n");
    printf("элементы которого равны количеству нулевых элементов в соответствующем столбце исходного массива\n\n\n");
    fflush(stdout);
    
    
    const int sideSize = Tools::getSizeFromInput();
    printf("Creating matrix...\n");
    fflush(stdout);
    const int nElements = sideSize * sideSize;
    const bool usePrint = sideSize <= 20;
    int *res = new int[sideSize];
    int *mat = new int[nElements];
    
    printf("Filling matrix...\n");
    fflush(stdout);
    Tools::fillArrayRandom(mat, nElements, -2, 2);

    printf("Start tests...\n");
    fflush(stdout);
    
    if (usePrint) {
        std::cout << "---------------" << std::endl;
        std::cout << "Initial Matrix:" << std::endl;
        std::cout << "---------------" << std::endl;
        Tools::printMatrix(mat, sideSize);
    }

    Tools::testFunctionCuda(Solve::cuda, Solve::testCuda, "CUDA", res, mat, sideSize, usePrint);
    //Tools::testFunction(Solve::testCuda, "CUDA TEST", res, mat, sideSize, usePrint);
    //Tools::testFunction(Solve::cuda, "CUDA", res, mat, sideSize, usePrint);
    Tools::testFunction(Solve::cpu, "CPU", res, mat, sideSize, usePrint);
    Tools::testFunction(Solve::cpuOpenMP, "CPU OpenMP", res, mat, sideSize, usePrint);
    
    // There is no way to use AVX256 in this case
    // So maby next time
    
    
    delete[] res;
    delete[] mat;

    //Cleanup cuda
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaDeviceReset failed!" << std::endl;
        _getch();
        return 1;
    }

    _getch();
    return 0;
}