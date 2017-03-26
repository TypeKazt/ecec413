# Final
### By: Alex Kazantsev and Hanumant Mahida
### Date: March 25th, 2017

Note: Theory questions are first, followed by the results from Q1-Q3.

## Theory Questions 

### Q1
 512 threads per block would yield the most if only 3 blocks are used, as 512*3 = 1536





### Q2
CUDA's current architechure has a wrap size of 32 threads, however there is no guarantee that in the future this will continue. Future changes would break the code. 

There are some solutions. One could use cudaGetDeviceProperties() to get the wrap size of the GPU and always use __syncthreads() when you have shared-memory dependencies between threads not in the same warp. 

One could also write code with 1 warp and not use any __syncthreads() by changing the warp_size -- but this could be limited to the application. 

Since __syncthreads() is a block wide barrier, it should be used to avoid shared memory race conditions. Avoiding its use does show significant performance gains, however it does race conditions and double-bufering becomes a worry.








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















## Results


## Timing data
| Matrix Size | CPU Time (s)  | CUDA Time (s) | Speed Improvement | 
|-------------|-----------------|-----------|----------------------|



## Code


```
