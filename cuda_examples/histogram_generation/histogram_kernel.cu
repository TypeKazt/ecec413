#ifndef _HISTOGRAM_KERNEL_H_
#define _HISTOGRAM_KERNEL_H_

__global__ void histogram_kernel_fast(int *input_data, int *histogram, int num_elements, int histogram_size)
{
    __shared__ unsigned int s[HISTOGRAM_SIZE];
	
    // Initialize the shared memory area 
    if(threadIdx.x < histogram_size)
        s[threadIdx.x] = 0;
		
    __syncthreads();

    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
	
    while(offset < num_elements){
        atomicAdd(&s[input_data[offset]], 1);
        offset += stride;
    }	  
	
    __syncthreads();

    if(threadIdx.x < histogram_size){
        atomicAdd(&histogram[threadIdx.x], s[threadIdx.x]);
    }
}


__global__ void histogram_kernel_slow(int *input_data, int *histogram, int num_elements, int histogram_size)
{
    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x; 
	
    while(offset < num_elements){
        atomicAdd(&histogram[input_data[offset]], 1);
        offset += stride;
    }	  
}

#endif // #ifndef _HISTOGRAM_KERNEL_H
