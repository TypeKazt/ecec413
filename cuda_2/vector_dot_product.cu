#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <float.h>

// includes, kernels
#include "vector_dot_product_kernel.cu"

void run_test(unsigned int);
float compute_on_device(float *, float *,int);
void check_for_error(char *);
extern "C" float compute_gold( float *, float *, unsigned int);

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
	float *A = (float *)malloc(size);
	float *B = (float *)malloc(size);
	
	// Randomly generate input data. Initialize the input data to be floating point values between [-.5 , 5]
	printf("Generating random vectors with values between [-.5, .5]. \n");	
	srand(time(NULL));
	for(unsigned int i = 0; i < num_elements; i++){
		A[i] = (float)rand()/(float)RAND_MAX - 0.5;
		B[i] = (float)rand()/(float)RAND_MAX - 0.5;
	}
	
	printf("Generating dot product on the CPU. \n");

	struct timeval start, stop;
	gettimeofday(&start, NULL);

	// Compute the reference solution on the CPU
	float reference = compute_gold(A, B, num_elements);

	gettimeofday(&stop, NULL);
	printf("Execution time = %fs. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));


	/* Edit this function to compute the result vector on the GPU. 
       The result should be placed in the gpu_result variable. */
	float gpu_result = compute_on_device(A, B, num_elements);

    /* Compare the CPU and GPU results. */
    float threshold = 0.001;
	printf("Result on CPU: %f, result on GPU: %f. \n", reference, gpu_result);
    if(fabsf((reference - gpu_result)/reference) < threshold){
        printf("TEST passed. \n");
    }
    else{
        printf("TEST failed. \n");
    }

	// cleanup memory
	free(A);
	free(B);
	
	return;
}

/* Edit this function to compute the dot product on the device. */
float compute_on_device(float *A_on_host, float *B_on_host, int num_elements)
{
	// Device vectors
	float *A_on_device = NULL;
	float *B_on_device = NULL;
	float *C_on_device = NULL;

	// allocate space and copy data to device for 3 vectors
	cudaMalloc((void**)&A_on_device, num_elements*sizeof(float));
	cudaMemcpy(A_on_device, A_on_host, num_elements*sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&B_on_device, num_elements*sizeof(float));
	cudaMemcpy(B_on_device, B_on_host, num_elements*sizeof(float), cudaMemcpyHostToDevice);
	
	cudaMalloc((void**)&C_on_device, GRID_SIZE*sizeof(float));
	cudaMemcpy(C_on_device, 0.0f, GRID_SIZE*sizeof(float));
	
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

	vector_dot_product_kernel <<< dimGrid, dimBlock >>> (num_elements, A_on_device, B_on_device, C_on_device, mutex);
	cudaThreadSynchronize();

	gettimeofday(&stop, NULL);
	printf("Execution time = %fs. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));

	check_for_error("Error in kernel");
	
	/* Copy first element of C_on_device to result */
	float result = 0.0f;
	cudaMemcpy( &result, C_on_device, sizeof(float), cudaMemcpyDeviceToHost);
	
	/* Free allocated vectors */
	cudaFree(A_on_device);
	cudaFree(B_on_device);
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
