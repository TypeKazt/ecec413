# Cuda Assignment #1
### By: Alex Kazantsev and Hanumant Mahida
### Date: March 6th, 2017

## Results
![CUDA Timing Graphs]
 (https://github.com/om23/ecec413/blob/master/cuda_1/cuda_timing_results.png)


## Code

###Cuda Global Memory

Load M and N onto device, allocate P on device
```C
Matrix Md = allocate_matrix_on_gpu(A);                  
	copy_matrix_to_device(Md, A);
        Matrix Nd = allocate_matrix_on_gpu(X);
        copy_matrix_to_device(Nd, X);

    Matrix Pd = allocate_matrix_on_gpu(Y); 
```
  
Setup execution grid
```C
dim3 threads(TILE_SIZE, TILE_SIZE);  
	dim3 grid((Pd.num_columns + TILE_SIZE - 1)/TILE_SIZE, (Pd.num_rows + TILE_SIZE - 1)/TILE_SIZE);
```

Execute the CUDA kernel
```C
vec_mat_kernel_naive<<< grid, threads >>>(Pd.elements, Md.elements, Nd.elements, MATRIX_SIZE);
        cudaThreadSynchronize();
```

###Cuda Shared Memory

Load and Allocate vector-matricies 
```C
    Matrix Ad = allocate_matrix_on_gpu(A);
    copy_matrix_to_device(Ad, A);
    Matrix Xd = allocate_matrix_on_gpu(X);
    copy_matrix_to_device(Xd, X);

    Matrix Yd = allocate_matrix_on_gpu(Y);
    copy_matrix_to_device(Yd, Y);
```

Execution Configuration
```C
    dim3 dimBlock, dimGrid;
    dimBlock.x = dimBlock.y = TILE_SIZE;
    dimBlock.z = 1;
    dimGrid.x = (Y.num_columns/ dimBlock.x) + ((Y.num_columns % dimBlock.x) ? 1:0 );
    dimGrid.y = (Y.num_rows / dimBlock.y) + ((Y.num_rows % dimBlock.y) ? 1:0 );
    dimGrid.z = 1;
```

Compute Device Computation Threads and Synchronize CUDA threads
```C
vec_mat_kernel_optimized<<<dimGrid,dimBlock>>>(Ad,Xd,Yd);
        cudaThreadSynchronize();

```


## Timing data
| Number of Elements | Serial Time (s)	| CUDA Global Time (s) | CUDA Shared Time (s) | Global Improvement | Shared Improvement | 
|-------------|-----------------|-------------------|--------------|--------------|--------------|
| 512  | 0.000903 | 0.000175 | 0.000117 | 5.16 | 7.72 |
| 1024 | 0.002815 | 0.000545 | 0.000367 | 5.17 | 7.67 |
| 2048 | 0.009133 | 0.001692 | 0.001248 | 5.40 | 7.32 |
