
#ifndef _2DCONVOLUTION_KERNEL_H_
#define _2DCONVOLUTION_KERNEL_H_

#include <stdio.h>
#include "2Dconvolution.h"

// __global__ void ConvolutionKernel(Matrix M, Matrix N, Matrix P, int matrix_size)
__global__ void ConvolutionKernel(Matrix M, Matrix N, Matrix P)

{

	// thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int x = tx + blockIdx.x * blockDim.x;
	int y = ty + blockIdx.y * blockDim.y;
	int tid = x + y * N.width;
	int KR = KERNEL_SIZE/2;
	int i, j;

	__shared__ float sN[BLOCK_SIZE + 4][BLOCK_SIZE + 4];

	i = x - KR; j = y - KR;
	if (i < 0 || j < 0)
		sN[ty][tx] = 0.f;
	else
		sN[ty][tx] = N.elements[tid - KR - KR * N.width];
	__syncthreads()

	i = x + KR; j = y - KR;
	if (i > N.width - 1 || j < 0)
	  sN[ty][tx + KR + KR] = 0.f;
	else
	  //sN[tx + KR + KR][ty] = 7.f;
	  sN[ty][tx + KR + KR] = N.elements[tid + KR - KR * N.width];
	__syncthreads();

	i = x - KR; j = y + KR;
	if (i < 0 || j > N.height - 1)
	  sN[ty + KR + KR][tx] = 0.f;
	else
	  //sN[tx][ty + KR + KR] = 7.f;
	  sN[ty + KR + KR][tx] = N.elements[tid - KR + KR * N.width];
	__syncthreads();

	i = x + KR; j = y + KR;
	if (i > N.width - 1 || j > N.height -1)
	  sN[ty + KR + KR][tx + KR + KR] = 0.f;
	else
	  //sN[tx + KR + KR][ty + KR + KR] = 7.f;
	  sN[ty + KR + KR][tx + KR + KR] = N.elements[tid + KR + KR * N.width];
	__syncthreads();


	float sum = 0.f;
	// convolute
	for (i = 0; i < KERNEL_SIZE; i ++)
		for (j = 0; j < KERNEL_SIZE; j++)
			sum += sN[ty + i][tx + j] * sM[i][j];

	if (tx < N.width && ty < N.height)
		P.elements[tid] = sum;

}

#endif // #ifndef _2DCONVOLUTION_KERNEL_H_
