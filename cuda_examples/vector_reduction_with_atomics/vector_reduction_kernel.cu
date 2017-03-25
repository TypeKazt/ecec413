#ifndef _VECTOR_REDUCTION_KERNEL_H_
#define _VECTOR_REDUCTION_KERNEL_H_

#define THREAD_BLOCK_SIZE 256           /* Size of a thread block. */
#define NUM_BLOCKS 240                  /* Number of thread blocks. */

__global__ void vector_reduction_kernel(float *A, float *C, unsigned int num_elements)
{
	__shared__ float sum_per_thread[THREAD_BLOCK_SIZE]; // Allocate shared memory to hold the partial sums.	
	unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;     // Obtain the thread ID.
	unsigned int stride = blockDim.x * gridDim.x; 
	double sum = 0.0f; 
	unsigned int i = thread_id; 

	/* Compute your partial sum. */
	while(i < num_elements){
		sum += (double)A[i];
		i += stride;
	}

	sum_per_thread[threadIdx.x] = (float)sum; // Copy sum to shared memory.
	__syncthreads(); // Wait for all threads in the thread block to finish up.

	/* Reduce the values generated by the thread block to a single value to be sent back to the CPU.
	   The following code assumes that the number of threads per block is power of two.
	 */
	i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i) 
			sum_per_thread[threadIdx.x] += sum_per_thread[threadIdx.x + i];
		__syncthreads();
		i /= 2;
	}

	/* Write the partial sum computed by this thread block to global memory. */
	if(threadIdx.x == 0)
		C[blockIdx.x] = sum_per_thread[0];
}


/* This function uses a compare and swap technique to acquire a mutex/lock. */
__device__ void lock(int *mutex)
{	  
    while(atomicCAS(mutex, 0, 1) != 0);
}

/* This function uses an atomic exchange operation to release the mutex/lock. */
__device__ void unlock(int *mutex)
{
    atomicExch(mutex, 0);
}


__global__ void vector_reduction_kernel_using_atomics(float *A, float *result, unsigned int num_elements, int *mutex)
{
	__shared__ float sum_per_thread[THREAD_BLOCK_SIZE];
	unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // Obtain the index of the thread.
	unsigned int stride = blockDim.x * gridDim.x; 
	double sum = 0.0f; 
	unsigned int i = thread_id; 

	/* Generate the partial sum. */
	while(i < num_elements){
		sum += (double)A[i];
		i += stride;
	}

	sum_per_thread[threadIdx.x] = (float)sum;          // Copy sum to shared memory.
	__syncthreads();                            // Wait for all thread in the thread block to finish.

	/* Reduce the values generated by the thread block to a single value. We assume that 
       the number of threads per block is power of two. */
	i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i) 
			sum_per_thread[threadIdx.x] += sum_per_thread[threadIdx.x + i];
		__syncthreads();
		i /= 2;
	}

	/* Accumulate the sum computed by this thread block into the global shared variable. */
	if(threadIdx.x == 0){
		lock(mutex);
		*result += sum_per_thread[0];
		unlock(mutex);
	}
}

#endif // #ifndef _VECTOR_REDUCTION_KERNEL_H
