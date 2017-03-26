 /* Device code. */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include "gauss_eliminate.h"

__global__ void gauss_eliminate_kernel(float *U, int k, int num_elements)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= k+1){
		U[num_elements * k + tid] = (float)(U[num_elements * k + tid] / U[num_elements * k + k]);
	}
	if (tid == k)
		U[num_elements*k+k] = 1;


	__syncthreads();

	if (tid >= k+1){
		for(int j = k+1; j < num_elements; j++)
		{
			U[num_elements * tid + j] -= U[num_elements * tid + k] * U[num_elements * k + j]; 
		}
	
		
		U[num_elements * tid + k] = 0;
	}
}


#endif // #ifndef _MATRIXMUL_KERNEL_H_
