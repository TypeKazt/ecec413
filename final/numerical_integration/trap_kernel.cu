/* Write GPU kernels to compete the functionality of estimating the integral via the trapezoidal rule. */ 

__global__ void integration(float *g_result, float a, int n, float h)
{
	extern __shared__ float sdata[]

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	float local_sum = 0;
	int stride = blockDim.x * gridDim.x;

	for(unsigned int j = i; j < n-1; j += stride)
		local_sum += f(a+k*h); 

	sdata[threadIdx.x] = local_sum;
	__syncthreads();


	for(unsigned int s=blockDim.x/2; s > 0; s>>=1)
	{
		if(tid < s)
		{
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	} 

	if(tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__device__ float 
f(float x) {
    return (x + 1)/sqrt(x*x + x + 1);
} 

