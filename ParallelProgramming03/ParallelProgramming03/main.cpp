#define NOMINMAX

#include <stdio.h>

#include <windows.h>
#include <conio.h>
#include <chrono>
#include <iostream>
#include <limits>

#include "Tools/tools.h"
#include "Sort/Cpu/cpu.h"
#include "Sort/Gpu/gpu.cuh"

#include "omp.h"

int main(int argc, char* argv[])
{
    //omp_set_nested(true);
    //if (argc > 1) {
    //    Tools::cmdMode(argc, argv); // Cmd mode for plotting
    //    return 0;
    //}

    SetConsoleCP(1251);
    SetConsoleOutputCP(1251);

    Tools::setupRandomizer();
    printf("----------\n");
    printf("CPU  info:\n");
    printf("----------\n");
    Tools::printCpuInfo();

    printf("\n\n\n");
    printf("Вариант № 11. Разработать программу для быстрой сортировки\n");
    fflush(stdout);
    
    
    const int64_t size = Tools::getSizeFromInput();
    const int64_t nIterations = Tools::getIterationsFromInput();
    printf("Creating matrix...\n");
    fflush(stdout);
    const bool usePrint = size <= 20;
    int64_t* initArray = nullptr;
    int64_t *array = nullptr;
    const size_t allocBytesSize = 2 * size * sizeof(int64_t);
    try {
        initArray = new int64_t[size];
        array = new int64_t[size];
    }
    catch (...)
    {
        std::cout << "Error: Can't allocate " << Tools::getMemoryAsString(allocBytesSize) << " (" << allocBytesSize << " B) of memory" << std::endl;
        _getch();
        return -1;
    }
    std::cout << "Memory allocated " << Tools::getMemoryAsString(allocBytesSize) << " (" << allocBytesSize << " B)" << std::endl;
    
    printf("Filling array...\n");
    fflush(stdout);
    Tools::fillArrayRandom(initArray, size, -100, 100);

    //First run
    Sort::cuda(array, 3);

    printf("Start tests...\n");
    fflush(stdout);
    
    if (usePrint) {
        std::cout << "---------------" << std::endl;
        std::cout << "Initial Array :" << std::endl;
        std::cout << "---------------" << std::endl;
        Tools::printArray(initArray, size);
    }

    Tools::testFunctionVerbose(Sort::multiThread, "CPU OpenMP", array, initArray, size, usePrint, nIterations, true);
    Tools::testFunctionVerbose(Sort::singleThread, "CPU", array, initArray, size, usePrint, nIterations, false);
    Tools::testFunctionVerbose(Sort::cudaSingle, "CUDA smart", array, initArray, size, usePrint, nIterations, false);
    Tools::testFunctionVerbose(Sort::cuda, "CUDA single", array, initArray, size, usePrint, nIterations, false);
    
    // There is no way to use AVX256 in this case
    // So maby next time
    
    
    delete[] array;
    delete[] initArray;

    _getch();
    return 0;
}