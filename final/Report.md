# Final
### By: Alex Kazantsev and Hanumant Mahida
### Date: March 25th, 2017

Note: Theory questions are first, followed by the results from Q1-Q3.

## Theory Questions 

### Q1
 512 threads per block would yield the most if only 3 blocks are used, as 512*3 = 1536





### Q2






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
