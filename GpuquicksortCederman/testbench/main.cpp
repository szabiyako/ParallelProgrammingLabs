#include "cuda_runtime_api.h"

#define _CRT_SECURE_NO_WARNINGS

#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "gpuqsort.h"
#include "defs.h"
#include "dists.h"
#include <string.h>

#include <conio.h>

typedef unsigned int element;


void testDists(char* uniid);
void testPhases(char* uniid);

#include "../src/simpletimer.cuh"


int main(int argc, char* argv[])
{
	testDists("");
	printf("Done\n");
	_getch();
}


/**
* Checks that data2 contains a sorted list of the elements in data
* @param data   An unsorted list of elements
* @param data2  The possibly correctly sorted version of data
* @param size   The size of the two lists
* @returns      True if data2 is the correctly sorted version of data
*/
bool validate(element* data, element* data2, unsigned int size)
{
	return true;
	// Sort data using a trusted method
	std::sort(data,data+size);

	// Compare each element to find any differences
	for(unsigned int i=0;i<size;i++)
		if(data[i]!=data2[i])
		{
			// data2 was not correctly sorted
			printf("Error at %d (%i != %i)!\n",i,data[i],data2[i]);
			return false;
		}

		// data2 was correctly sorted
		return true;
}

int saveResults(int testsize, const char* distribution, float time, int threads, int maxblocks, int sbsize, char* uniid, unsigned int phase, unsigned int test)
{	
	printf("ArraySize: %d DistributionType: %s Time: %fms\n", testsize, distribution, time);
	return 0;
}

/**
* Tries all distributions ITERATIONS times
*/
void testDists(char* uniid)
{
	const unsigned int MEASURES = 5;
	const unsigned int DISTRIBUTIONS = 6;
	const unsigned int STARTSIZE = 2<<19;

	// Allocate memory for the sequences to be sorted
	unsigned int maxsize = STARTSIZE<<(MEASURES-1);
	element* data = new element[maxsize];
	element* data2 = new element[maxsize];

	double timerValue;
	unsigned int run = 0;

	// Go through all distributions
	for(int d=0;d<DISTRIBUTIONS;d++)
	{
		unsigned int testsize = STARTSIZE;

		// Go through all sizes
		for(int i=0;i<MEASURES;i++,testsize<<=1)
		{
			// Do it several times
			for(int q=0;q<ITERATIONS;q++)
			{
				// Create sequence according to distribution
				dist(data,testsize,d);
				// Store copy of sequence
				memcpy(data2,data,testsize*sizeof(element));

				int threads  =0;
				int maxblocks=0;
				int sbsize   =0;

				if(gpuqsort(data,testsize,&timerValue,maxblocks,threads,sbsize,0)!=0)
				{
					printf("Error! (%s)\n",getGPUSortErrorStr());
					exit(1);
				}

				// Validate the result
				if(!validate(data2,data,testsize))
					exit(1); 

				saveResults(testsize,getDistString(d),(float)timerValue,threads,maxblocks,sbsize,uniid,0,1);
				printf("%d/%d\n",++run,MEASURES*DISTRIBUTIONS*ITERATIONS);
			}
		}
	}
}

/**
* Tries different combinations of parameters and measures each phase
*/
void testPhases(char* uniid)
{
	unsigned int testsize = 2<<22;
	element* data = new element[testsize];
	element* data2 = new element[testsize];
	double timerValue;

	unsigned int run=0;

	// Uses same distribution for all
	int d = 0;
	dist(data2,testsize,d);

	// Test different sizes
	for(testsize = 2<<18;testsize<=(2<<22)+500;testsize *= 2)  // 5
	{
		// Measure each phase
		for(int phase=0;phase<3;phase++)  // 3
			// Vary the number of threads
			for(int threads=32;threads<=256;threads*=2) // 4
				// Vary the number of blocks
				for(int maxblocks=32;maxblocks<=1024;maxblocks*=2)  // 6
					// Vary when to switch to bitonic
					for(int sbsize=64;sbsize<=2048;sbsize*=2) // 6
						// Do it several times
						for(int q=0;q<ITERATIONS;q++)
						{
							// Store a copy sequence for reuse
							memcpy(data,data2,testsize*4);

							// Sort it
							if(gpuqsort(data,testsize,&timerValue,maxblocks,threads,sbsize,phase)!=0)
							{
								printf("Error! (%s)\n",getGPUSortErrorStr());
								exit(1);
							}

							saveResults(testsize,getDistString(d),(float)timerValue,threads,maxblocks,sbsize,uniid,phase,0);
							printf("%d/%d!\n",run++,2160*ITERATIONS);
						}
	}
}


