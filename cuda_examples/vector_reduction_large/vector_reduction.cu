/* Reduction of arbitrary sized vectors. Host side code. This is perhaps not the best implementation since 
   the size of the problem directly influences the number of threads created. 
   Author: Naga Kandasamy
   Date: 2/23/2017
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <float.h>

// includes, kernels
#include "vector_reduction_kernel.cu"

#define NUM_ELEMENTS 50000000 // Keep in mind the limit on grid dimensions

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);
float computeOnDevice(float* h_data, int array_mem_size);

extern "C" 
void computeGold( float* reference, float* idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int 
main( int argc, char** argv) 
{
	runTest( argc, argv);
	return 1;
}


/* Perform the reduction on the GPU. */
void
runTest( int argc, char** argv) 
{
	int num_elements = NUM_ELEMENTS;


	// -- Parse Command Line -------------------------------------------
	if (argc != 1)
		num_elements = atoi(argv[1]);
	// -----------------------------------------------------------------


	// -- Generate Input Data ------------------------------------------
	size_t array_mem_size = sizeof( float) * num_elements;
	float* h_data = (float*) malloc( array_mem_size);
	
	srand(time(NULL)); // Seed the pseudo-random number generator
	printf ("\nNum Elements: %i\n\n", num_elements);
	for( unsigned int i = 0; i < num_elements; ++i) 
		h_data[i] = floorf(2*(rand()/(float)RAND_MAX));

	// -- Compute CPU Solution -----------------------------------------
	float reference = 0.0f; 	
	printf("Performing reduction on the CPU...");
	computeGold(&reference , h_data, num_elements);
    printf("done\n");

	// -- Compute GPU Solution -----------------------------------------
	printf("Performing reduction on the GPU.\n");
	float result = computeOnDevice(h_data, num_elements);
	// -----------------------------------------------------------------

	// -- Check the GPU Results ----------------------------------------
	float epsilon = 0.0f;
	unsigned int result_regtest = (abs(result - reference) <= epsilon);
	printf( "Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");
	printf( "device: %f\n", result);
	printf( "host: %f\n", reference);
	// -----------------------------------------------------------------

	free( h_data);
}



float computeOnDevice(float* h_data, int num_elements)
{
	float* gpu_data;
	size_t gpu_data_size = num_elements * sizeof (float);

	// -- Allocate GPU Memory & Copy Data Over -------------------------
	cudaMalloc ((void**)&gpu_data, gpu_data_size);
	cudaMemcpy (gpu_data, h_data, gpu_data_size, cudaMemcpyHostToDevice);

	// -- Determine GPU Execution Configuration ------------------------
	int threads_per_block = 512;
	int thread_blocks = (int)ceil((float)num_elements/(float)threads_per_block);
	int grid_dimension_in_blocks = (int)ceil(sqrt((float)thread_blocks)); // Set up a 2D grid

	// -- Define GPU Execution Config ----------------------------------
	dim3 dimGrid  (grid_dimension_in_blocks, grid_dimension_in_blocks, 1);
	dim3 dimBlock (threads_per_block, 1, 1);
	int smemSize = threads_per_block * sizeof(float);

	// -- Perform the first set of partial sum reductions -------------------------------
    int iter = 1;
	printf("Iteration %d: Performing the first set of reductions on the input data.\n", iter);
	printf("Using a grid of %d x %d thread blocks to process %d elements. \n", grid_dimension_in_blocks, grid_dimension_in_blocks, num_elements);
	reduction <<<dimGrid, dimBlock, smemSize>>> (gpu_data, num_elements);
	// cudaThreadSynchronize();

	// -- Perform more reductions if needed---------
    iter++;
	printf("\nPerforming additional reductions on the GPU.\n");
	int num_partial_sums = thread_blocks;
	while(num_partial_sums > threads_per_block){
		// Determine the number of thread blocks needed to process this set of partial sums
		thread_blocks = (int)ceil((float)num_partial_sums/(float)threads_per_block);
		
		// Set up the 2D grid 
        grid_dimension_in_blocks = (int)ceil(sqrt((float)thread_blocks)); // Set up a 2D grid
		dim3 dimGrid  (grid_dimension_in_blocks, grid_dimension_in_blocks, 1);
		dim3 dimBlock (threads_per_block, 1, 1);
		smemSize = threads_per_block * sizeof(float);
        printf("Iteration %d: using a grid of %d x %d thread blocks to process %d elements. \n", iter, grid_dimension_in_blocks, grid_dimension_in_blocks, num_partial_sums);
	
		// Perform the reduction 
		reduction <<<dimGrid, dimBlock, smemSize>>> (gpu_data, num_partial_sums);
		// cudaThreadSynchronize();

		// Update the number of partial sums
		num_partial_sums = thread_blocks;
		iter++;
	}

	// -- Perform the final reduction ------------------------------
	printf("\n Iteration %d: performing the final reduction on %d elements. \n", iter, num_partial_sums);
	if(num_partial_sums <= threads_per_block){ 
		dim3 dimGrid (1, 1, 1);
		dim3 dimBlock (512, 1, 1);
		smemSize = 512 * sizeof (float);
		reduction <<<dimGrid, dimBlock, smemSize>>> (gpu_data, num_partial_sums);
	}

	// -- Copy Result to CPU Memory ------------------------------------
	cudaMemcpy (h_data, gpu_data, sizeof(float), cudaMemcpyDeviceToHost);

	// -- Free the GPU Memory ------------------------------------------
	cudaFree (gpu_data); 
	// -----------------------------------------------------------------

	return h_data[0];
}
