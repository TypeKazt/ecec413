/* Host side code. 
   Reduction of arbitrary sized vectors using atomics. 
   Also shows the use of pinned memory to map a portion of the CPU address space to the GPU's address space.

   Author: Naga Kandasamy
   Date: 02/14/2017
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

/* includes, kernels. */
#include "vector_reduction_kernel.cu"

/* Declaration, forward. */
void run_test(unsigned int);
void run_test_with_pinned_memory(unsigned int);
float compute_on_device(float *, int);
float compute_on_device_with_pinned_memory(float *, int);
void check_for_error(const char *);
extern "C" float compute_gold( float *, unsigned int);

int 
main( int argc, char** argv) 
{
    if(argc != 2){
		printf("Usage: vector_reduction <num elements> \n");
		exit(0);	
	}

	/* Check the device properties if our device supports mapping host memory. */
	cudaDeviceProp properties;
	int my_device;
	cudaGetDevice(&my_device);
	cudaGetDeviceProperties(&properties, my_device);
	if(properties.canMapHostMemory != 1){
			  printf("The device cannot map host memory. \n");
			  exit(0);
	}

	/* Place the CUDA runtime in a state which supports mapping memory on the host. */ 
	cudaSetDeviceFlags(cudaDeviceMapHost);

	unsigned int num_elements = atoi(argv[1]);

	run_test(num_elements);
	printf("\n");
	run_test_with_pinned_memory(num_elements);

	return 0;
}

void            /* Perform reduction on the CPU and the GPU and compare results for correctness. */ 
run_test(unsigned int num_elements) 
{
	/* Obtain the vector length. */
	unsigned int vector_size = sizeof(float) * num_elements;

	/* Allocate memory on the CPU for the input vector A. */
	float *A = (float *)malloc(vector_size);
		
	/* Randomly generate input data. Initialize the input data to be values between -.5 and +.5. */	
	printf("Creating a random vector with %d elements.\n", num_elements);
	srand(time(NULL));
	for(unsigned int i = 0; i < num_elements; i++)
		A[i] = (float)rand()/(float)RAND_MAX - .5;
		
	/* Reduce the vector on the CPU. */
	printf("Reducing the vector with %d elements on the CPU.\n", num_elements);
	float reference = compute_gold(A, num_elements);
	printf("CPU result using double precision arithmetic: %f \n", reference);

	/* Compute the result vector on the GPU. */ 
	printf("Reducing the vector with %d elements on the GPU.\n", num_elements);
	float gpu_result = compute_on_device(A, num_elements);
	printf("GPU result using double precision arithmetic: %f \n", gpu_result);
	
	/* cleanup memory. */
	free(A);
	return;
}

/* Perform vector reduction on the CPU and the GPU and compare results for correctness. 
   Use pinned memory. 
 */
void 
run_test_with_pinned_memory(unsigned int num_elements) 
{
	/* Obtain the vector length. */
	unsigned int vector_size = sizeof(float) * num_elements;

	/* Allocate pinned memory on the CPU for the input vector A. */
	float *A = NULL;
	cudaHostAlloc((void **)&A, vector_size, cudaHostAllocWriteCombined | cudaHostAllocMapped);
	check_for_error("ERROR ALLOCATING PINNED MEMORY");

	/* Randomly generate input data. Initialize the input data to be integer values between -.5 and +.5. 
	   Since the writes to A are combined, read/write operations on them are not guranteed to be coherent
	   That is, the writes to A are not guaranteed to be visible until a fence operation is called. 
	   In our case, that will happen when the GPU kernel is called.
	 */
	printf("Creating a random vector with %d elements.\n", num_elements);
	srand(time(NULL));
	for(unsigned int i = 0; i < num_elements; i++)
		A[i] = (float)rand()/(float)RAND_MAX - 0.5;

	/* Reduce the vector on the CPU. */
	printf("Reducing the vector with %d elements on the CPU.\n", num_elements);

	float reference = compute_gold(A, num_elements);

	printf("CPU result using double precision arithmetic: %f \n", reference);


	/* Compute the result vector on the GPU. */ 
	printf("Reducing the vector with %d elements on the GPU. Using pinned memory to hold the vector.\n", num_elements);

	
	float gpu_result = compute_on_device_with_pinned_memory(A, num_elements);

	printf("GPU result using double precision arithmetic: %f \n", gpu_result);

	/* cleanup memory. */
	cudaFreeHost(A);
	check_for_error("ERROR FREEING PINNED MEMORY");
	return;
}


float 
compute_on_device(float *A_on_host, int num_elements)
{
	float *A_on_device = NULL;
	float *result_on_device = NULL;

	/* Allocate space on the GPU for vector A and copy the contents to the GPU. */
	cudaMalloc((void**)&A_on_device, num_elements * sizeof(float));
	cudaMemcpy(A_on_device, A_on_host, num_elements * sizeof(float), cudaMemcpyHostToDevice);

	/* Allocate space for the result on the GPU and initialize it. */
	cudaMalloc((void**)&result_on_device, sizeof(float));
	cudaMemset(result_on_device, 0.0f, sizeof(float));

	/* Allocate space for the lock on the GPU and initialize it. */
	int *mutex_on_device = NULL;
	cudaMalloc((void **)&mutex_on_device, sizeof(int));
	cudaMemset(mutex_on_device, 0, sizeof(int));

 	/* Set up the execution grid on the GPU. */
	dim3 thread_block(THREAD_BLOCK_SIZE, 1, 1); 
	dim3 grid(NUM_BLOCKS,1);
	
	/* Launch the kernel. */
	vector_reduction_kernel_using_atomics<<<grid, thread_block>>>(A_on_device, result_on_device, num_elements, mutex_on_device);
	cudaThreadSynchronize();

	
	check_for_error("KERNEL FAILURE");

	float sum;
	cudaMemcpy(&sum, result_on_device, sizeof(float), cudaMemcpyDeviceToHost);

	/* Free memory. */
	cudaFree(A_on_device);
	cudaFree(result_on_device);
	cudaFree(mutex_on_device);

	return sum;
}

float 
compute_on_device_with_pinned_memory(float *A_on_host, int num_elements)
{
	float *A_on_device = NULL;

	/* Get valid GPU pointers using pointers on the pinned memory on the host. */ 
	cudaHostGetDevicePointer(&A_on_device, A_on_host, 0);
	check_for_error("ERROR GETTING GPU POINTER");

	/* Allocate pinned memory on the CPU for the result. */
	float *result_on_host = NULL;
	float *result_on_device = NULL;
	cudaHostAlloc((void **)&result_on_host, sizeof(float) * NUM_BLOCKS, cudaHostAllocMapped);
	cudaHostGetDevicePointer(&result_on_device, result_on_host, 0);
	check_for_error("ERROR ALLOCATING PINNED MEMORY");

 	/* Set up the execution grid on the GPU. */
	dim3 thread_block(THREAD_BLOCK_SIZE, 1, 1); // Set the number of threads in the thread block
	dim3 grid(NUM_BLOCKS,1);
	
	/* Launch the kernel. */
	vector_reduction_kernel<<<grid, thread_block>>>(A_on_device, result_on_device, num_elements);
	cudaThreadSynchronize();
	check_for_error("KERNEL FAILURE");

	/* Perform final set of reductions on the CPU. */
	double sum = 0.0f;
	for(unsigned int i = 0; i < NUM_BLOCKS; i++)
		sum += result_on_host[i];

	/* Free memory. */
	cudaFreeHost(result_on_host);
	check_for_error("ERROR FREEING PINNED MEMORY");

	return (float)sum;
}


void 
check_for_error(const char *msg){
	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err){
		printf("CUDA ERROR: %s (%s). \n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
} 
