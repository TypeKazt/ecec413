/* Matrix multiplication: P = M * N. Device code. 

Author: Naga Kandasamy
Date modified: 11/19/2014

*/

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include "matrixmul.h"

// Declare 1D textures to hold the M and N matrices
texture<float> M_on_tex;
texture<float> N_on_tex;

// Declare 2D textures to hold the M and N matrices
texture<float, 2> M_on_tex_2D;
texture<float, 2> N_on_tex_2D;


/* Example of a kernel that uses 2D textures. */
__global__ void 
MatrixMulKernel_2Dtex(float* P, const float* M, const float* N, int matrix_size){
	
	// Thread index
	int threadX = threadIdx.x; 
    int threadY = threadIdx.y;

	// Block index
	int blockX = blockIdx.x; 
    int blockY = blockIdx.y;

	// Find position in Matrix
	int column_number = TILE_SIZE * blockX + threadX; 
    int row_number = TILE_SIZE * blockY + threadY;

	double P_temp = 0;
	
    for (int k = 0; k < matrix_size; k++) {
		
        /* Scan through row elements. Texture values are indexed in (x, y), that is (col, row) form rather 
           than the (y, x) or (row, col) form. */
        double M_element = tex2D(M_on_tex_2D, k, row_number); 
        double N_element = tex2D(N_on_tex_2D, column_number, k);;
		
        P_temp += M_element * N_element; 
	}
	
	// Write result to P
	P[row_number * matrix_size + column_number] = (float)P_temp;
}


/* Example of a kernel that uses a 1D texture. */
__global__ void 
MatrixMulKernel_1Dtex(float* P, const float* M, const float* N, int matrix_size)
{
	// Thread index
	int threadX = threadIdx.x; 
    int threadY = threadIdx.y;

	// Block index
	int blockX = blockIdx.x; 
    int blockY = blockIdx.y;

	// Find position in Matrix
	int column_number = TILE_SIZE * blockX + threadX; 
    int row_number = TILE_SIZE * blockY + threadY;

	double P_temp = 0;
	
    for (int k = 0; k < matrix_size; k++) {
		
        double M_element = tex1Dfetch(M_on_tex, (matrix_size * row_number + k)); // Scan through row elements
		double N_element = tex1Dfetch(N_on_tex, (matrix_size * k + column_number));;
		
        P_temp += M_element * N_element; 
	}
	
	// Write result to P
	P[row_number * matrix_size + column_number] = (float)P_temp;
}


__global__ void 
MatrixMulKernel_vanilla(float* P, const float* M, const float* N, int matrix_size)
{
	// Thread index
	int threadX = threadIdx.x; 
    int threadY = threadIdx.y;

	// Block index
	int blockX = blockIdx.x; 
    int blockY = blockIdx.y;

	// Find position in Matrix
	int column_number = TILE_SIZE * blockX + threadX; 
    int row_number = TILE_SIZE * blockY + threadY;

	double P_temp = 0;
	for (int k = 0; k < matrix_size; k++) {
		double M_element = M[matrix_size * row_number + k]; // Scan through row elements
		double N_element = N[matrix_size * k + column_number];
		P_temp += M_element * N_element; 
	}
	
	// Write result to P
	P[row_number * matrix_size + column_number] = (float)P_temp;
}

#endif
