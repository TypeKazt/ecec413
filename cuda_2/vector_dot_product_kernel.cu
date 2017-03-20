#ifndef _VECTOR_DOT_PRODUCT_KERNEL_H_
#define _VECTOR_DOT_PRODUCT_KERNEL_H_

#define BLOCK_SIZE 1024
#define GRID_SIZE 1024

/* Edit this function to complete the functionality of dot product on the GPU. 
	You may add other kernel functions as you deem necessary. 
 */

__device__ void lock(int *mutex);
__device__ void unlock(int *mutex);

__global__ void vector_dot_product_kernel(int num_elements, float* a, float* b, float* result, int *mutex)
{
	__shared__ float runningSums[BLOCK_SIZE];

	int tx = threadIdx.x;
	int threadID = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	float local_thread_sum = 0.0;
	unsigned int i = threadID;

	while(i < num_elements){
		local_thread_sum += a[i] * b[i];
		i += stride;
	}

	runningSums[threadIdx.x] = local_thread_sum;
	__syncthreads();

	for(int stride = blockDim.x/2; stride > 0; stride /= 2){
		if(tx < stride)
			runningSums[tx] += runningSums[tx+stride];
		__syncthreads();
	}

	if(threadIdx.x == 0) {
		lock(mutex);
		result[0] += runningSums[0];
		unlock(mutex);
	}
}


__device__ void lock(int *mutex){
	while(atomicCAS(mutex, 0, 1) != 0);
}

__device__ void unlock(int *mutex){
	atomicExch(mutex, 0);
}


#endif // #ifndef _VECTOR_DOT_PRODUCT_KERNEL_H
