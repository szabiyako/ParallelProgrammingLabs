#include "cuda_runtime_api.h"


#ifdef _MSC_VER
#include <windows.h>
#else
#include <sys/time.h>
#endif

class SimpleTimer
{
#ifdef _MSC_VER
	LARGE_INTEGER starttime;
#else
	struct timeval starttime;
#endif
public:
	void start();
	double end(); 
};

