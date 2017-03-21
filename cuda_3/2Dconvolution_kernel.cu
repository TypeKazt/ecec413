
#ifndef _2DCONVOLUTION_KERNEL_H_
#define _2DCONVOLUTION_KERNEL_H_

#include <stdio.h>
#include "2Dconvolution.h"

// __global__ void ConvolutionKernel(Matrix M, Matrix N, Matrix P, int matrix_size)
__constant__ float sM[KERNEL_SIZE][KERNEL_SIZE];

__global__ void ConvolutionKernel(Matrix N, Matrix P, int num_elements)

{

    // int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    // int idx = threadId;
    // int i, j, m, n, pointi, pointj;
    // double sum;
    // int len = N.width;

    // while (idx < num_elements){
    //     sum = 0;
    //     i = idx/len;
    //     j = idx%len;
    //     for (m = 0; m < 5; m++){
    //         for (n = 0; n < 5; n++){
    //             pointi = i+m-2;
    //             pointj = j+n-2;
    //             if (!(pointi < 0 || pointj < 0 || pointi >= len || pointj >= len)){
    //                 sum+= kernel_c[m * 5 + n] * N.elements[pointi*len+pointj];
    //             }
    //         }
    //     }
    //     P.elements[idx] = (float)sum;
    //     idx += 2048;
    // }

    // return;

	// thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = tx + blockIdx.x * blockDim.x;
  int y = ty + blockIdx.y * blockDim.y;
  int tid = x + y * N.width;
  int KR = KERNEL_SIZE/2;
  int i, j;

  // Load M into constant memory
  /*__constant__ float sM[KERNEL_SIZE][KERNEL_SIZE];
  if (x < KERNEL_SIZE && y < KERNEL_SIZE)
    sM[y][x] = M.elements[x + y * M.width];*/

  __shared__ float sN[THREAD_BLOCK_SIZE + 4][THREAD_BLOCK_SIZE  + 4];

  // Handle 4 corner cases of P
  i = x - KR; j = y - KR;
  if (i < 0 || j < 0)
    sN[ty][tx] = 0.f;
  else
    //sN[tx][ty] = 7.f;
    sN[ty][tx] = N.elements[tid - KR - KR * N.width];
  __syncthreads();
  
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
  // Convolute
  for (i = 0; i < KERNEL_SIZE; i++)
    for (j = 0; j < KERNEL_SIZE; j++)
      sum += sN[ty + i][tx + j] * sM[i][j];

  if (tx < N.width && ty < N.height)
    P.elements[tid] = sum;

}

#endif // #ifndef _2DCONVOLUTION_KERNEL_H_
