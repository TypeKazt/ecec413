#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>


// includes, kernels
#include "scan_naive_kernel.cu"

void runTest( int argc, char** argv);
extern "C" unsigned int compare( const float* reference, const float* data, const unsigned int len);
extern "C" void computeGold( float* reference, float* idata, const unsigned int len);
void checkCUDAError(const char *msg);
int checkResults(float *, float *, int, float);

int 
main( int argc, char** argv) 
{
    runTest( argc, argv);
    exit(0);
}

void
runTest( int argc, char** argv) 
{ 
    unsigned int num_elements = 512;
    const unsigned int mem_size = sizeof( float) * num_elements;


    const unsigned int shared_mem_size = sizeof(float) * num_elements;

    // allocate host memory to store the input data
    float* h_data = (float*) malloc(mem_size);
      
    // initialize the input data on the host to be integer values
    // between 0 and 10
    for( unsigned int i = 0; i < num_elements; ++i){
        h_data[i] = floorf(10*(rand()/(float)RAND_MAX));
    }

    // compute reference solution
    float* reference = (float*) malloc( mem_size);  
    computeGold( reference, h_data, num_elements);

    // allocate device memory input and output arrays
    float* d_idata;
    float* d_odata;
    cudaMalloc( (void**) &d_idata, mem_size);
    cudaMalloc( (void**) &d_odata, mem_size);

    // copy host memory to device input array
    cudaMemcpy( d_idata, h_data, mem_size, cudaMemcpyHostToDevice);

    // setup execution parameters
    // Note that these scans only support a single thread-block worth of data,
    dim3  grid(1, 1, 1);
    dim3 threads(512, 1, 1);
 
    printf("Running parallel prefix sum (scan) of %d elements\n", num_elements);

    scan_naive<<< grid, threads, 2 * shared_mem_size >>>(d_odata, d_idata, num_elements);
    cudaThreadSynchronize();

        
    // copy result from device to host
    cudaMemcpy( h_data, d_odata, sizeof(float) * num_elements, cudaMemcpyDeviceToHost);
        
    float epsilon = 0.0f;
    unsigned int result_regtest = checkResults( reference, h_data, num_elements, epsilon);
    printf( "Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");

    // cleanup memory
    free( h_data);
    free( reference);
    cudaFree(d_idata);
    cudaFree(d_odata);
}

void 
checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		printf("CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}						 
}


int 
checkResults(float *reference, float *gpu_result, int num_elements, float threshold)
{
    int checkMark = 1;
    for(int i = 0; i < num_elements; i++)
        if((reference[i] - gpu_result[i]) > threshold){
            checkMark = 0;
            break;
        }

    return checkMark;
}
