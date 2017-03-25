#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <float.h>

// includes, kernels
#include "trap_kernel.cu"
#define LEFT_ENDPOINT 10
#define RIGHT_ENDPOINT 1005
#define NUM_TRAPEZOIDS 100000000



void run_test(unsigned int);
float compute_on_device(float , float ,int, float);
void check_for_error(char *);
extern "C" float compute_gold( float , float , int, float);

int main( int argc, char** argv) 
{
	if(argc != 2){
		printf("Usage: vector_dot_product <num elements> \n");
		exit(0);	
	}
	unsigned int num_elements = atoi(argv[1]);
	run_test(num_elements);
	return 0;
}

/* Perform vector dot product on the CPU and the GPU and compare results for correctness.  */
void run_test(unsigned int num_elements) 
{
	// Obtain the vector length
	unsigned int size = sizeof(float) * num_elements;

	// Allocate memory on the CPU for the input vectors A and B
	int n = NUM_TRAPEZOIDS;
	float a = LEFT_ENDPOINT;
	float b = RIGHT_ENDPOINT;
	float h = (b-a)/(float)n; // Height of each trapezoid  
	
	// Randomly generate input data. Initialize the input data to be floating point values between [-.5 , 5]
/*	printf("Generating random vectors with values between [-.5, .5]. \n");	
	srand(time(NULL));
	for(unsigned int i = 0; i < num_elements; i++){
		A[i] = (float)rand()/(float)RAND_MAX - 0.5;
		B[i] = (float)rand()/(float)RAND_MAX - 0.5;
	}
	
*/	printf("Generating intergral on CPU. \n");

	struct timeval start, stop;
	gettimeofday(&start, NULL);

	// Compute the reference solution on the CPU
	float reference = compute_gold(a, b, n, h);

	gettimeofday(&stop, NULL);
	printf("Execution time = %fs. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));


	/* Edit this function to compute the result vector on the GPU. 
       The result should be placed in the gpu_result variable. */
	float gpu_result = compute_on_device(a, b, n, h);

    /* Compare the CPU and GPU results. */
    float threshold = 0.001;
	printf("Result on CPU: %f, result on GPU: %f. \n", reference, gpu_result);
    if(fabsf((reference - gpu_result)/reference) < threshold){
        printf("TEST passed. \n");
    }
    else{
        printf("TEST failed. \n");
    }

	return;
}

/* Edit this function to compute the dot product on the device. */
float compute_on_device(float A, float B, int num_elements, float h)
{
	// Device vectors
	float *C_on_device = NULL;

	// allocate space and copy data to device for 3 vectors

	cudaMalloc((void**)&C_on_device, GRID_SIZE*sizeof(float));
	cudaMemset(C_on_device, 0.0, GRID_SIZE*sizeof(float));
	
	// device mutex, add to each threadblock reduction
	int *mutex = NULL;
	cudaMalloc((void **)&mutex, sizeof(int));
	cudaMemset(mutex, 0, sizeof(int));

	// Thread block and grid inits
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(GRID_SIZE, 1);

	struct timeval start, stop;

	printf("performing vector dot product on GPU. \n");
	gettimeofday(&start, NULL);

	vector_dot_product_kernel <<< dimGrid, dimBlock >>> (num_elements, A, B, C_on_device, mutex);
	cudaThreadSynchronize();

	gettimeofday(&stop, NULL);
	printf("Execution time = %fs. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));

	check_for_error("Error in kernel");
	
	/* Copy first element of C_on_device to result */
	float result = 0.0;
	cudaMemcpy( &result, C_on_device, sizeof(float), cudaMemcpyDeviceToHost);
	result *= h;
	
	/* Free allocated vectors */
	cudaFree(C_on_device);
	
	return result;

}
 
/* This function checks for errors returned by the CUDA run time. */
void check_for_error(char *msg){
	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err){
		printf("CUDA ERROR: %s (%s). \n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
} 
