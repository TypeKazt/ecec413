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
#define NUM_TRAPEZOIDS 10



void run_test();
float compute_on_device(float , float ,int, float);
void check_for_error(char *);
//extern "C" float compute_gold( float , float , int, float);

float 
fab(float x) {
    return (x + 1)/sqrt(x*x + x + 1);
}

float 
f(float x) {
    return (x + 1)/sqrt(x*x + x + 1);
}  /* f */

/*------------------------------------------------------------------
 * Function:    Trap
 * Purpose:     Estimate integral from a to b of f using trap rule and
 *              n trapezoids
 * Input args:  a, b, n, h
 * Return val:  Estimate of the integral 
 */
double 
compute_gold(float a, float b, int n, float h) {
   double integral;
   int k;

   integral = (f(a) + f(b))/2.0;
   for (k = 1; k <= n-1; k++) {
     integral += f(a+k*h);
   }
   integral = integral*h;

   return integral;
}

int main( int argc, char** argv) 
{
	run_test();
	return 0;
}

/* Perform vector dot product on the CPU and the GPU and compare results for correctness.  */
void run_test() 
{

	// Allocate memory on the CPU for the input vectors A and B
	int n = NUM_TRAPEZOIDS;
	float a = LEFT_ENDPOINT;
	float b = RIGHT_ENDPOINT;
	float h = (b-a)/(float)n; // Height of each trapezoid  
	
	
	printf("Generating intergral on CPU 100mil traps. \n");

	struct timeval start, stop;
	gettimeofday(&start, NULL);

	// Compute the reference solution on the CPU
	double reference = compute_gold(a, b, NUM_TRAPEZOIDS, h);

	gettimeofday(&stop, NULL);
	printf("Execution time = %fs. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));


	/* Edit this function to compute the result vector on the GPU. 
       The result should be placed in the gpu_result variable. */
	double gpu_result = compute_on_device(a, b, NUM_TRAPEZOIDS, h);

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

	// allocate space and copy data to device for 1 vectors

	cudaMalloc((void**)&C_on_device, GRID_SIZE*sizeof(float));
	cudaMemset(C_on_device, 0.0, GRID_SIZE*sizeof(float));
	
	// device mutex, add to each threadblock reduction
	int *mutex = NULL;
	cudaMalloc((void **)&mutex, sizeof(int));
	cudaMemset(mutex, 0, sizeof(int));

	// Thread block and grid inits
	dim3 dimBlock(1000, 1, 1);
	dim3 dimGrid(65535, 100000000-65535*BLOCK_SIZE);

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
	double pre =  (fab(A) + fab(B))/2.0f; 
	printf("pre: %f\n", pre);
	printf("pre result: %f \n", result);
	printf("h: %f \n", h);
	result += pre;
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
