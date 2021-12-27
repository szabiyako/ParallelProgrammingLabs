#include "tools.h"

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <chrono>
#include <sstream>
#include <limits>

#include "omp.h"

#include "InstructionSet/InstructionSet.h"

using namespace Tools;

static std::string floatToString(const float value, const int precision = 1)
{
    std::ostringstream out;
    out.precision(precision);
    out << std::fixed << value;
    return out.str();
}

void Tools::printCpuInfo()
{
    auto& outstream = std::cout;

    auto support_message = [&outstream](std::string isa_feature, bool is_supported) {
        //std::cout << std::setw(25) << tokenAsString(current.token) << std::setw(5);
        outstream << std::setw(13) << std::left << isa_feature << std::right << std::setw(5) << (is_supported ? "supported" : "-") << std::endl;
    };

    std::cout << InstructionSet::Vendor() << std::endl;
    std::cout << InstructionSet::Brand() << std::endl;

    support_message("3DNOW", InstructionSet::_3DNOW());
    support_message("3DNOWEXT", InstructionSet::_3DNOWEXT());
    support_message("ABM", InstructionSet::ABM());
    support_message("ADX", InstructionSet::ADX());
    support_message("AES", InstructionSet::AES());
    support_message("AVX", InstructionSet::AVX());
    support_message("AVX2", InstructionSet::AVX2());
    support_message("AVX512CD", InstructionSet::AVX512CD());
    support_message("AVX512ER", InstructionSet::AVX512ER());
    support_message("AVX512F", InstructionSet::AVX512F());
    support_message("AVX512PF", InstructionSet::AVX512PF());
    support_message("BMI1", InstructionSet::BMI1());
    support_message("BMI2", InstructionSet::BMI2());
    support_message("CLFSH", InstructionSet::CLFSH());
    support_message("CMPXCHG16B", InstructionSet::CMPXCHG16B());
    support_message("CX8", InstructionSet::CX8());
    support_message("ERMS", InstructionSet::ERMS());
    support_message("F16C", InstructionSet::F16C());
    support_message("FMA", InstructionSet::FMA());
    support_message("FSGSBASE", InstructionSet::FSGSBASE());
    support_message("FXSR", InstructionSet::FXSR());
    support_message("HLE", InstructionSet::HLE());
    support_message("INVPCID", InstructionSet::INVPCID());
    support_message("LAHF", InstructionSet::LAHF());
    support_message("LZCNT", InstructionSet::LZCNT());
    support_message("MMX", InstructionSet::MMX());
    support_message("MMXEXT", InstructionSet::MMXEXT());
    support_message("MONITOR", InstructionSet::MONITOR());
    support_message("MOVBE", InstructionSet::MOVBE());
    support_message("MSR", InstructionSet::MSR());
    support_message("OSXSAVE", InstructionSet::OSXSAVE());
    support_message("PCLMULQDQ", InstructionSet::PCLMULQDQ());
    support_message("POPCNT", InstructionSet::POPCNT());
    support_message("PREFETCHWT1", InstructionSet::PREFETCHWT1());
    support_message("RDRAND", InstructionSet::RDRAND());
    support_message("RDSEED", InstructionSet::RDSEED());
    support_message("RDTSCP", InstructionSet::RDTSCP());
    support_message("RTM", InstructionSet::RTM());
    support_message("SEP", InstructionSet::SEP());
    support_message("SHA", InstructionSet::SHA());
    support_message("SSE", InstructionSet::SSE());
    support_message("SSE2", InstructionSet::SSE2());
    support_message("SSE3", InstructionSet::SSE3());
    support_message("SSE4.1", InstructionSet::SSE41());
    support_message("SSE4.2", InstructionSet::SSE42());
    support_message("SSE4a", InstructionSet::SSE4a());
    support_message("SSSE3", InstructionSet::SSSE3());
    support_message("SYSCALL", InstructionSet::SYSCALL());
    support_message("TBM", InstructionSet::TBM());
    support_message("XOP", InstructionSet::XOP());
    support_message("XSAVE", InstructionSet::XSAVE());
}

void Tools::printMatrix(char *arr, const int64_t sideSize)
{
    for (int64_t row = 0; row < sideSize; ++row) {
        std::cout << "|";
        for (int64_t col = 0; col < sideSize; ++col) {
            const int64_t elemIdx = col * sideSize + row;
            std::cout.width(4);
            std::cout << int(arr[elemIdx]);
            std::cout << " ";
        }
        std::cout << "\b|" << std::endl;
    }
}

void Tools::printArray(int64_t *arr, const int64_t size)
{
    const int64_t last = size - 1;
    std::cout << "[";
    for (int64_t i = 0; i < last; ++i) {
        std::cout << arr[i] << ", ";
    }
    std::cout << arr[last] << "]" << std::endl;
    fflush(stdout);
}

int64_t Tools::getSizeFromInput()
{
    const int64_t maxN = 370727;
    int64_t N = -1;
    while ((N <= 0) || (N > maxN)) {
        while (std::cout << "Enter N = " && !(std::cin >> N)) {
            std::cin.clear(); //clear bad input flag
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); //discard input
            std::cout << "Invalid input\n";
        }
        if (N <= 0)
            std::cout << "Invalid input\n";
        else if (N > maxN)
            std::cout << "N = " << N << " is greater than maxN = " << maxN << "\n";
    }
    return N;
}

int64_t Tools::getIterationsFromInput()
{
    const int64_t maxIters = 100;
    int64_t result = 1;
    std::string input;
    bool ok = false;
    std::getline(std::cin, input);
    while (!ok) {
        std::cout << "Enter Iters (or press Enter to set default 20) = ";
        std::getline(std::cin, input);
        if (input.length() == 0) {
            return 20;
        }
        int64_t iters;
        try {
            iters = std::stoll(input);
        }
        catch (...) {
            std::cout << "Invalid input\n";
        }
        if (iters <= 0)
            std::cout << "Invalid input\n";
        else if (iters > maxIters)
            std::cout << "Iters = " << iters << " is greater than maxIters = " << maxIters << "\n";
        else {
            result = iters;
            ok = true;
        }
    }

    return result;
}

void Tools::fillArrayRandom(char *arr, const int64_t size, const char minValue, const char maxValue)
{ 
    #pragma omp parallel
    {
        srand(uint32_t(time(NULL)) ^ omp_get_thread_num());
        #pragma omp for
        for (int64_t i = 0; i < size; ++i) {
            arr[i] = getRandomChar(minValue, maxValue);
        }
    }
}

void Tools::clearArray(int64_t *arr, const int64_t size)
{
    for (int64_t i = 0; i < size; ++i) {
        arr[i] = 0;
    }
}

void Tools::setupRandomizer()
{
    srand(uint32_t(time(NULL)));
}

char Tools::getRandomChar(const char min, const char max)
{
    const int32_t range = std::abs(max - min) + 1;
    return (rand() % range) + min;
}

void Tools::testFunctionVerbose(
    void (*function)(int64_t*, const char*, const int64_t),
    const char* name,
    int64_t* result,
    const char* arr,
    const int64_t size,
    const bool usePrint,
    const int64_t iterations)
{
    //Clear result
    clearArray(result, size);

    size_t minTimeNs = 0;
    size_t maxTimeNs = 0;
    size_t avgTimeNs = 0;

    testFunction(
        function,
        result,
        arr,
        size,
        iterations,
        avgTimeNs,
        maxTimeNs,
        minTimeNs);
    printf("--------------\n");
    printf(" %s:\n", name);
    printf("--------------\n");
    if (usePrint)
        printArray(result, size);
    fflush(stdout);
    std::cout << "Average time: " << getConvertedTime(avgTimeNs) << std::endl;
    std::cout << "    Max time: " << getConvertedTime(maxTimeNs) << std::endl;
    std::cout << "    Min time: " << getConvertedTime(minTimeNs) << std::endl;
}

void Tools::testFunction(
    void (*function)(int64_t*, const char*, const int64_t),
    int64_t* result,
    const char* arr,
    const int64_t size,
    const int64_t iterations,
    size_t& avgTime,
    size_t& maxTime,
    size_t& minTime)
{
    size_t minTimeNs = std::numeric_limits<size_t>::max();
    size_t maxTimeNs = std::numeric_limits<size_t>::min();
    size_t avgTimeNs = 0;

    for (int64_t i = 0; i < iterations; ++i) {
        //Clear result
        clearArray(result, size);

        const std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();
        function(result, arr, size);
        const std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
        const size_t timeInNs = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count();
        if (timeInNs < minTimeNs)
            minTimeNs = timeInNs;
        if (timeInNs > maxTimeNs)
            maxTimeNs = timeInNs;
        avgTimeNs += timeInNs;
    }

    avgTime = avgTimeNs / iterations;
    maxTime = maxTimeNs;
    minTime = minTimeNs;
}

std::string Tools::getConvertedTime(const size_t timeInNs)
{
    const size_t timeInMs = timeInNs / 1000000ull;
    const size_t timeInSeconds = timeInMs / 1000;
    const size_t timeInMinutes = timeInSeconds / 60;
    const size_t timeInHours = timeInMinutes / 60;
    const size_t ns = timeInNs % 1000000ull;
    const size_t ms = timeInMs % 1000;
    const size_t sec = timeInSeconds % 60;
    const size_t min = timeInMinutes % 60;
    const size_t hrs = timeInHours;

    std::string result;
    if (hrs > 0)
        result += std::to_string(hrs) + "h, ";
    if (min > 0)
        result += std::to_string(min) + "min, ";
    if (sec > 0)
        result += std::to_string(sec) + "sec, ";
    if (ms > 0)
        result += std::to_string(ms) + "ms, ";
    result += std::to_string(ns) + "ns";

    return result;
}

std::string Tools::getMemoryAsString(const size_t bytes)
{
    const float kbytes = (float(bytes)) / 1024;
    const float mbytes = (float(bytes)) / (std::pow(1024ll, 2ll));
    const float gbytes = (float(bytes)) / (std::pow(1024ll, 3ll));
    const float tbytes = (float(bytes)) / (std::pow(1024ll, 4ll));

    if (tbytes > 1.f)
        return floatToString(tbytes) + " TiB";
    if (gbytes > 1.f)
        return floatToString(gbytes) + " GiB";
    if (mbytes > 1.f)
        return floatToString(mbytes) + " MiB";
    if (kbytes > 1.f)
        return floatToString(kbytes) + " KiB";
    return std::to_string(bytes) + " B";
}

#include "../Solve/Cpu/solveCpu.h"
#include "../Solve/CpuOpenMP/solveCpuOpenMP.h"

void Tools::cmdMode(int argc, char* argv[])
{
    if (argc != 2)
        exit(-1);

    enum Type : int
    {
        CPU,
        STATIC,
        DYNAMIC,
        GUIDED
    };

    const std::string maxSideStr = argv[1];
   
    int64_t maxSide = 0;

    try {
        maxSide = std::stoll(maxSideStr);
    }
    catch (...) {
        exit(-1);
    }

    int64_t* res = nullptr;
    char* mat = nullptr;
    try {
        res = new int64_t[maxSide];
        mat = new char[maxSide * maxSide];
    }
    catch (...)
    {
        exit(-1);
    }

    ////////////////////////////////////////////////////

    size_t avgTime = 0.f;
    size_t minTime = 0.f;
    size_t maxTime = 0.f;

    Tools::fillArrayRandom(mat, maxSide * maxSide, -2, 2);

    while (true) {
        int type = 0;
        std::cin >> type;
        if ((type < 0) || (type > 3))
            exit(0);

        int64_t inputSideSize;
        int64_t inputIterations;
        std::cin >> inputSideSize >> inputIterations;

        const int64_t sideSize = inputSideSize;
        const int64_t nIterations = inputIterations;
        const int64_t nElements = sideSize * sideSize;

        if (type == CPU)
            Tools::testFunction(Solve::cpu, res, mat, sideSize, nIterations, avgTime, maxTime, minTime);
        else if (type == STATIC)
            Tools::testFunction(Solve::cpuOpenMpStatic, res, mat, sideSize, nIterations, avgTime, maxTime, minTime);
        else if (type == DYNAMIC)
            Tools::testFunction(Solve::cpuOpenMpDynamic, res, mat, sideSize, nIterations, avgTime, maxTime, minTime);
        else if (type == GUIDED)
            Tools::testFunction(Solve::cpuOpenMpGuided, res, mat, sideSize, nIterations, avgTime, maxTime, minTime);
        else
            exit(-1);

        std::cout << avgTime << " " << maxTime << " " << minTime;
    }

    

    delete[] res;
    delete[] mat;

    exit(0);
}
