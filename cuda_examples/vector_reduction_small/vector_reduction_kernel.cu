
/* 
	Device code for vector reduction. 

	Author: Naga Kandasamy
	Date modified: 02/14/2017

 */

#ifndef _SCAN_NAIVE_KERNEL_H_
#define _SCAN_NAIVE_KERNEL_H_

#define NUM_ELEMENTS 512

// This kernel performs reduction using a tree-style reduction technique that increases divergent branching between threads in a warp
__global__ void reduction_v1(float *g_data, int n)
{
	__shared__ float partialSum[NUM_ELEMENTS];

	// Find our place in thread block/grid
	unsigned int threadID = threadIdx.x;
	unsigned int dataID = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Populate shared memory with data from global memory 
	if(dataID < n) 
		partialSum[threadID] = g_data[dataID];
	else
		partialSum[threadID] = 0.0;
 
	__syncthreads();

	// Calculate partial sum
	for (unsigned int stride = 1; stride < blockDim.x; stride *= 2){
		if (threadID % (2 * stride) == 0)
			partialSum[threadID] += partialSum[threadID + stride];
		__syncthreads();
	}

	// Store result in the appropriate place in the output stream
	if (threadID == 0)
		g_data[blockIdx.x] = partialSum[0];
}

// This kernel performs reduction in a fashion that reduces divergent branching between threads in a warp
__global__ void reduction_v2(float *g_data, int n)
{
	__shared__ float partialSum[NUM_ELEMENTS];
	// Find our place in thread block/grid 
	unsigned int threadID = threadIdx.x; 
	unsigned int dataID = blockIdx.x * blockDim.x + threadIdx.x; 

	// Copy data to shared memory from global memory 
	if(dataID < n) 
		partialSum[threadID] = g_data[dataID];
	else
		partialSum[threadID] = 0.0;
	
	__syncthreads();

	for(unsigned int stride = blockDim.x >> 1; stride > 0; stride = stride >> 1){
		if(threadID < stride)
			partialSum[threadID] += partialSum[threadID + stride];
	
		__syncthreads();
	}

	// Store result in the appropriate place in the output stream
	if(threadID == 0)
		g_data[blockIdx.x] = partialSum[0];
}

#endif // #ifndef _SCAN_NAIVE_KERNEL_H_
