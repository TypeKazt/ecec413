
#ifndef _2DCONVOLUTION_KERNEL_H_
#define _2DCONVOLUTION_KERNEL_H_

#include <stdio.h>
#include "2Dconvolution.h"

// __global__ void ConvolutionKernel(Matrix M, Matrix N, Matrix P, int matrix_size)
__global__ void ConvolutionKernel(const float* M, const float* N, float* P, int matrix_size)

{

	// thread index
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;

	// block index
	int blockX = blockIdx.x;
	int blockY = blockIdx.y;

	// find position in matrix
	int column_number = blockDim.x * blockX + threadX;
	int row_number = blockDim.y * blockY + threadY;

	double P_temp = 0;
	for (int k = 0; k < matrix_size; k++) {
		double M_element = M[matrix_size * row_number + k]; 
		double N_element = N[matrix_size * k + column_number];
		P_temp += M_element * N_element;
	}

	// write result to P
	P[row_number * matrix_size + column_number] = (float)P_temp;

}

#endif // #ifndef _2DCONVOLUTION_KERNEL_H_
