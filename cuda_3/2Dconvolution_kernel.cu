
#ifndef _2DCONVOLUTION_KERNEL_H_
#define _2DCONVOLUTION_KERNEL_H_

#include <stdio.h>
#include "2Dconvolution.h"

__global__ void ConvolutionKernel(Matrix N, Matrix P)

{

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = threadId;
    int i, j, m, n, point_i, point_j;
    double sum;

    while (idx < num_elements){
        sum = 0;
        i = idx/N.width;
        j = idx%N.width;
        for (m = 0; m < 5; m++){
            for (n = 0; n < 5; n++){
                point_i = i+m-2;
                point_j = j+n-2;
                if (!(point_i < 0 || point_j < 0 || point_i >= N.width || point_j >= N.width)){
                    sum+= kernel_c[m * 5 + n] * N.elements[point_i*N.width+point_j];
                }
            }
        }
        P.elements[idx] = (float)sum;
        idx += 2048;
    }

    return;

	// // thread index
 //  int tx = threadIdx.x;
 //  int ty = threadIdx.y;
 //  int x = tx + blockIdx.x * blockDim.x;
 //  int y = ty + blockIdx.y * blockDim.y;
 //  int tid = x + y * N.width;
 //  int KR = KERNEL_SIZE/2;
 //  int i, j;

 //  // Load M into constant memory
 //  /*__constant__ float sM[KERNEL_SIZE][KERNEL_SIZE];
 //  if (x < KERNEL_SIZE && y < KERNEL_SIZE)
 //    sM[y][x] = M.elements[x + y * M.width];*/

 //  __shared__ float sN[THREAD_BLOCK_SIZE + 4][THREAD_BLOCK_SIZE  + 4];

 //  // Handle 4 corner cases of P
 //  i = x - KR; j = y - KR;
 //  if (i < 0 || j < 0)
 //    sN[ty][tx] = 0.f;
 //  else
 //    //sN[tx][ty] = 7.f;
 //    sN[ty][tx] = N.elements[tid - KR - KR * N.width];
 //  __syncthreads();
  
 //  i = x + KR; j = y - KR;
 //  if (i > N.width - 1 || j < 0)
 //    sN[ty][tx + KR + KR] = 0.f;
 //  else
 //    //sN[tx + KR + KR][ty] = 7.f;
 //    sN[ty][tx + KR + KR] = N.elements[tid + KR - KR * N.width];
 //  __syncthreads();

 //  i = x - KR; j = y + KR;
 //  if (i < 0 || j > N.height - 1)
 //    sN[ty + KR + KR][tx] = 0.f;
 //  else
 //    //sN[tx][ty + KR + KR] = 7.f;
 //    sN[ty + KR + KR][tx] = N.elements[tid - KR + KR * N.width];
 //  __syncthreads();

 //  i = x + KR; j = y + KR;
 //  if (i > N.width - 1 || j > N.height -1)
 //    sN[ty + KR + KR][tx + KR + KR] = 0.f;
 //  else
 //    //sN[tx + KR + KR][ty + KR + KR] = 7.f;
 //    sN[ty + KR + KR][tx + KR + KR] = N.elements[tid + KR + KR * N.width];
 //  __syncthreads();

 //  float sum = 0.f;
 //  // Convolute
 //  for (i = 0; i < KERNEL_SIZE; i++)
 //    for (j = 0; j < KERNEL_SIZE; j++)
 //      sum += sN[ty + i][tx + j] * sM[i][j];

 //  if (tx < N.width && ty < N.height)
 //    P.elements[tid] = sum;

}

#endif // #ifndef _2DCONVOLUTION_KERNEL_H_
