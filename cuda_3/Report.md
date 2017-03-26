# Cuda Assignment #3
### By: Alex Kazantsev and Hanumant Mahida
### Date: March 19th, 2017

## Results
![CUDA Timing Graphs]
(https://raw.githubusercontent.com/om23/ecec413/master/cuda_3/timingdata.png)


## Timing data
| Matrix Size | CPU Time (s)  | CUDA Time (s) | Speed Improvement | 
|-------------|-----------------|-----------|----------------------|
| 512 | 0.026786 | 0.000351 | 76.31339031 |
| 1024 | 0.100798 | 0.001256 | 80.25318471 |
| 2048 | 2.331 | 0.001587 | 66.62295417 |


## Code

Load M to device
```C
Matrix Nd = AllocateDeviceMatrix(N);
CopyToDeviceMatrix(Nd, N);
```

Allocate P on device
```C
Matrix Pd = AllocateDeviceMatrix(P);
CopyToDeviceMatrix(Pd, P); // Clear memory
```

Setup Execution Config
```C
dim3 grid(THREAD_BLOCK_SIZE);
dim3 thread(THREAD_BLOCK_SIZE*2);
cudaMemcpyToSymbol(kernel_c, M.elements, 25*sizeof(float)); 
```

Launch Device Comp threads
```C
ConvolutionKernel<<<grid, thread>>>(Nd, Pd);
cudaThreadSynchronize();
```

Kernel
```C

int threadId = blockIdx.x * blockDim.x + threadIdx.x;
int idx = threadId;
int i, j, m, n, point_i, point_j;
double sum;

while (idx < NUM_ELEMENTS){
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

```
