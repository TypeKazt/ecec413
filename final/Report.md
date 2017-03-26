# Final
### By: Alex Kazantsev and Hanumant Mahida
### Date: March 25th, 2017

Note: Result summaries from Q1-Q3, then Theory questions are first, followed by the code from Q1-Q3.


## Programing Question 1 (Gauss)




## Programming Question 2 (Trap)





## Programming Question 3 (Compact Stream)

The results for Compact Stream on the GPU showed about 4x-5x *slower* performance than when running compact stream on the CPU. 

For example, for 1024 elements CPU time was 0.000015 seconds and GPU time was 0.000050 seconds.


## Theory Questions 

### Q1
 512 threads per block would yield the most if only 3 blocks are used, as 512*3 = 1536





### Q2
CUDA's current architechure has a wrap size of 32 threads, however there is no guarantee that in the future this will continue. Future changes would break the code. 

There are some solutions. One could use cudaGetDeviceProperties() to get the wrap size of the GPU and always use __syncthreads() when you have shared-memory dependencies between threads not in the same warp. 

One could also write code with 1 warp and not use any __syncthreads() by changing the warp_size -- but this could be limited to the application. 

Since __syncthreads() is a block wide barrier, it should be used to avoid shared memory race conditions. Avoiding its use does show significant per\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\formance gains, however it does race conditions and double-bufering becomes a worry.








### Q3

a) 
36 FLOP/ 200 GFLOPS = 1.8e-10 s
7B*4B/100GB = 2.8e-10 s
memory bound

b)
36 FLOP/ 300 GFLOPS = 1.2e-10 s
7B*4B/250GB = 1.12e-10 s
compute bound


### Q4

The final sum will be skewed due to floating point precision error that is an artifact of floating point addition. To correct for precision error the Kahan summation algorithm must be leveraged.





### Q5








## Code

### Programming Question 1 (Gauss)



### Programming Question 2 (Trap)



### Programming Question 3 (Compact Stream)
Allocate vectors onto the device
```C
cudaMalloc((void**)&result_device, num_elements*sizeof(float));
cudaMemcpy(result_device, result_d, num_elements*sizeof(float), cudaMemcpyHostToDevice);

cudaMalloc((void**)&h_device, num_elements*sizeof(float));
cudaMemcpy(h_device, h_data, num_elements*sizeof(float), cudaMemcpyHostToDevice);
```

Thread Block and Grid inits
```C
dim3 threads(TILE_SIZE, TILE_SIZE);
dim3 grid(num_elements);
```

Execute Kernel and Sync Threads
```C
compact_stream_kernel<<<grid, threads>>>(result_device, h_device, num_elements, n);
cudaThreadSynchronize();
```
Kernel
```C
for (unsigned int i = 0; i < len; i++) {
    if (idata[idx] > 0.0) {
        //reference[n++] = idata[idx];
        *n++;
}
```