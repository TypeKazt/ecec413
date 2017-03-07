/* Vector reduction example using shared memory. WOrks for small vectors.
Author: Naga Kandasamy
Date modified: 02/14/2017
*/


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

// includes, kernels
#include "vector_reduction_kernel.cu"

// For simplicity, just to get the idea in this MP, we're fixing the problem size to 512 elements.
#define NUM_ELEMENTS 512

void runTest( int argc, char** argv);
float computeOnDevice(float* h_data, int array_mem_size);
void checkCUDAError(const char *msg);
extern "C" void computeGold( float* reference, float* idata, const unsigned int len);

int 
main( int argc, char** argv) 
{
	runTest( argc, argv);
	return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
//! Run naive scan test
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
	int num_elements = NUM_ELEMENTS;
	const unsigned int array_mem_size = sizeof( float) * num_elements;

	// allocate host memory to store the input data
	float* h_data = (float*) malloc(array_mem_size);

	// initialize the input data on the host to be integer values
	// between 0 and 1000
	srand(time(NULL));
	for(unsigned int i = 0; i < num_elements; ++i){
		h_data[i] = floorf(1000*(rand()/(float)RAND_MAX));
	}
	// compute reference solution
	float reference = 0.0f;  
	computeGold(&reference , h_data, num_elements);
	
	float result = computeOnDevice(h_data, num_elements);


	// We can use an epsilon of 0 since values are integral and in a range 
	// that can be exactly represented
	float epsilon = 0.0f;
	unsigned int result_regtest = (abs(result - reference) <= epsilon);
	printf( "Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");
	printf( "device: %f  host: %f\n", result, reference);
	// cleanup memory
	free( h_data);
}

// Take h_data from host, copies it to device, setup grid and thread 
// dimentions, excutes kernel function, and copy result of scan back
// to h_data.
// Note: float* h_data is both the input and the output of this fun/*
float 
computeOnDevice(float* h_data, int num_elements)
{
	// initialize variables for device data, cuda error and timer
	float* d_data;

	int data_size = sizeof(float) * num_elements;

	// allocate memory on device
	cudaMalloc((void**)&d_data, data_size);
	checkCUDAError("Error allocating memory");

	// copy host memory to device
	cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice);
	checkCUDAError("Error copying host to device memory");

	// Invoke kernel
	int threads = 256;
	int blocks = 2;
	reduction_v1<<<blocks, threads>>>(d_data, num_elements);
	// reduction_v2<<<blocks, threads>>>(d_data, num_elements);
	checkCUDAError("Error in kernel");

	// copy device memory to host
	cudaMemcpy(h_data, d_data, data_size, cudaMemcpyDeviceToHost);
	checkCUDAError("Error copying host to device memory");

	// cleanup device memory
	cudaFree(d_data);
	checkCUDAError("Error freeing memory");


	// calculate final result of two partially calculated blocks
	float result = h_data[0] + h_data[1];

	return result;
}

void 
checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) 
	{
		printf("CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}						 
}
