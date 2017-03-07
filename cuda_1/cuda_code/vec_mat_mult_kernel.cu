/* Vector-Matrix multiplication: Y = A * X.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include "vec_mat_mult.h"

/* Write the kernel for vector-matrix multiplication using GPU global memory. */
__global__ void vec_mat_kernel_naive(float* P, const float* M, const float* N, int matrix_size)
{
	//Multiply A and X
	// Thread index
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;

	// Block index
	int blockX = blockIdx.x;
	int blockY = blockIdx.y;

	// Find position in Matrix
	int column_number = blockDim.x * blockX + threadX;
	int row_number = blockDim.y * blockY + threadY;

	double P_temp = 0;
	for (int k = 0; k < matrix_size; k++) {
		double M_element = M[matrix_size * row_number + k]; // Scan through row elements
		//double N_element = N[matrix_size * k + column_number];
		double N_element = N[k];
		P_temp += M_element * N_element; 
	}

	// Write result to P
	//P[row_number * matrix_size + column_number] = (float)P_temp;
	P[row_number] = (float)P_temp;
	//P[0] = (float)P_temp;
}


/* Write the kernel for vector-matrix multiplication using GPU shared memory. */
__global__ void vec_mat_kernel_optimized(Matrix M, Matrix N, Matrix P)
{
	//Multiply A and X

    __shared__ float Msub[TILE_SIZE][TILE_SIZE];
    __shared__ float Nsub[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x; // Obtain the x-index within the thread block
    int ty = threadIdx.y; // Obtain the y-index within the thread block
    int row = (blockDim.y * blockIdx.y + ty); // Perform the thread to data ID mapping
    int col = blockDim.x * blockIdx.x + tx;
    int k = 0;
    int temp;
    double Psub = 0.0f;
  
    while(k < M.num_columns){
        // Check M edge condtions for this tile
        if(k + tx < M.num_columns && row < M.num_rows)
            Msub[ty][tx] = M.elements[row*M.num_columns + k + tx];
        else
            Msub[ty][tx] = 0.0f; // Pad out the shared memory area 

    
        // Check N edge conditions for this tile
        if(k + threadIdx.y < N.num_rows && col < N.num_columns)
            Nsub[ty][tx] = N.elements[(k+ty)*N.num_columns + col];
        else
            Nsub[ty][tx] = 0.0f; // Pad out the shared memory area

        __syncthreads(); // Barrier sync for threads to wait while shared memory is populated by the thread block

    
        // Multiply the row and column entries corresponding to the tile just loaded 
        for(temp = 0; temp < TILE_SIZE; temp++)
            Psub += Msub[ty][temp] * Nsub[temp][tx];

        __syncthreads();
    
        k += TILE_SIZE;
  }

    // Output edge condition check
    if(col < P.num_columns && row < P.num_rows)
        P.elements[row*P.num_columns + col] = (float)Psub;

    return;

}



#endif 
