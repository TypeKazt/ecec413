# Cuda Assignment #2
### By: Alex Kazantsev and Hanumant Mahida
### Date: March 19th, 2017

## Results
![CUDA Timing Graphs]
(https://github.com/om23/ecec413/blob/master/cuda_2/timingdata.png)


## Timing data
| Number of Elements | Serial Time (s)  | CUDA Time (s) | Speed Improvement | 
|-------------|-----------------|-----------|----------------------|
| 10e5 | 0.001847 | 0.002176 | 0.849 |
| 10e6 | 0.017127 | 0.000471 | 36.36 |
| 10e7 | 0.142393 | 0.003285 | 43.35 |


## Code



Allocate memory and copy data to device for 3 vectors
```C
cudaMalloc((void**)&A_on_device, num_elements*sizeof(float));
cudaMemcpy(A_on_device, A_on_host, num_elements*sizeof(float), cudaMemcpyHostToDevice);

cudaMalloc((void**)&B_on_device, num_elements*sizeof(float));
cudaMemcpy(B_on_device, B_on_host, num_elements*sizeof(float), cudaMemcpyHostToDevice);

cudaMalloc((void**)&C_on_device, GRID_SIZE*sizeof(float));
cudaMemset(C_on_device, 0.0, GRID_SIZE*sizeof(float));
```


Performing vector dot product on GPU
```C
vector_dot_product_kernel <<< dimGrid, dimBlock >>> (num_elements, A_on_device, B_on_device, C_on_device, mutex);
cudaThreadSynchronize();
```


Vector Dot Product Kernel for handling mutex locks and unlocks
```C
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
```


