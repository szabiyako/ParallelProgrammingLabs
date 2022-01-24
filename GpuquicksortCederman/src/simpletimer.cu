#include "simpletimer.cuh"

#include "cuda_runtime_api.h"

#ifdef _MSC_VER
void SimpleTimer::start()
{
	QueryPerformanceCounter(&starttime);
}

double SimpleTimer::end()
{
	LARGE_INTEGER endtime, freq;
	QueryPerformanceCounter(&endtime);
	QueryPerformanceFrequency(&freq);

	return ((double)(endtime.QuadPart - starttime.QuadPart)) / ((double)(freq.QuadPart / 1000.0));
}
#else
void SimpleTimer::start()
{
	gettimeofday(&starttime, 0);
}

double SimpleTimer::end()
{
	struct timeval endtime;
	gettimeofday(&endtime, 0);

	return (endtime.tv_sec - starttime.tv_sec) * 1000.0 + (endtime.tv_usec - starttime.tv_usec) / 1000.0;
}
#endif