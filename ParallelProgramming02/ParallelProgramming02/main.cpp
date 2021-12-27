#define NOMINMAX

#include <stdio.h>

#include <windows.h>
#include <conio.h>
#include <chrono>
#include <iostream>
#include <limits>

#include "Tools/tools.h"
#include "Solve/Cpu/solveCpu.h"
#include "Solve/CpuOpenMP/solveCpuOpenMP.h"

int main(int argc, char* argv[])
{
    if (argc > 1) {
        Tools::cmdMode(argc, argv); // Cmd mode for plotting
        return 0;
    }
    SetConsoleCP(1251);
    SetConsoleOutputCP(1251);

    Tools::setupRandomizer();
    printf("----------\n");
    printf("CPU  info:\n");
    printf("----------\n");
    Tools::printCpuInfo();

    printf("\n\n\n");
    printf("Вариант № 11. Написать программу с использованием технологии OPENMP, которая реализует следующие действия:\n");
    printf("формирует массив, размерностью N x N и формирует новый массив,\n");
    printf("элементы которого равны количеству нулевых элементов в соответствующем столбце исходного массива\n\n\n");
    fflush(stdout);
    
    
    const int64_t sideSize = Tools::getSizeFromInput();
    const int64_t nIterations = Tools::getIterationsFromInput();
    printf("Creating matrix...\n");
    fflush(stdout);
    const int64_t nElements = sideSize * sideSize;
    const bool usePrint = sideSize <= 20;
    int64_t* res = nullptr;
    char* mat = nullptr;
    const size_t allocBytesSize = nElements * sizeof(char) + sideSize * sizeof(int64_t);
    try {
        res = new int64_t[sideSize];
        mat = new char[nElements];
    }
    catch (...)
    {
        std::cout << "Error: Can't allocate " << Tools::getMemoryAsString(allocBytesSize) << " (" << allocBytesSize << " B) of memory" << std::endl;
        _getch();
        return -1;
    }
    std::cout << "Memory allocated " << Tools::getMemoryAsString(allocBytesSize) << " (" << allocBytesSize << " B)" << std::endl;
    
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

    Tools::testFunctionVerbose(Solve::cpu, "CPU", res, mat, sideSize, usePrint, nIterations);
    Tools::testFunctionVerbose(Solve::cpuOpenMpStatic, "CPU OpenMP Static", res, mat, sideSize, usePrint, nIterations);
    Tools::testFunctionVerbose(Solve::cpuOpenMpDynamic, "CPU OpenMP Dynamic", res, mat, sideSize, usePrint, nIterations);
    Tools::testFunctionVerbose(Solve::cpuOpenMpGuided, "CPU OpenMP Guided", res, mat, sideSize, usePrint, nIterations);
    
    // There is no way to use AVX256 in this case
    // So maby next time
    
    
    delete[] res;
    delete[] mat;

    _getch();
    return 0;
}