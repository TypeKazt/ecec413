#ifndef _REDUCTION_KERNEL_H_
#define _REDUCTION_KERNEL_H_

__global__ void reduction(float *g_data, int n)
{
	// -- Get pointer to shared memory ---------------------------------
	extern __shared__ float s_data[];
	// -----------------------------------------------------------------

	// -- Determine the thread's position in the 2D grid -----------------------
	int threadIdxInGrid = blockIdx.y * gridDim.x * blockDim.x + 
			      blockDim.x * blockIdx.x + 
			      threadIdx.x;
	// -----------------------------------------------------------------

	// -- Populate Shared Memory ---------------------------------------
	if (threadIdxInGrid >= n)
		s_data[threadIdx.x] = 0.0;
	else
		s_data[threadIdx.x] = g_data[threadIdxInGrid];
	// -----------------------------------------------------------------
	
	__syncthreads();
	
	// -- Perform Sum Reduction ----------------------------------------
	for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
		if (threadIdx.x < s)
			s_data[threadIdx.x] += s_data[threadIdx.x + s];
	
		__syncthreads();
	}
	// -----------------------------------------------------------------

	// -- Copy Result from Shared Mem to Global Mem --------------------
	if (threadIdx.x == 0)
		g_data[blockIdx.y * gridDim.x + blockIdx.x] = s_data[0];
	// -----------------------------------------------------------------
}

#endif // #ifndef _REDUCTION_KERNEL_H_
