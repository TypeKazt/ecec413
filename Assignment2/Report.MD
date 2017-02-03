# Report of Applying OpenMP to Histograms

### By: Alex Kazantsev and Hanumant Mahida
### Date: February 2th, 2017

## Implimentation & Results
The most optimal technique to parallelize the histogram program was to impliment an OpenMP Pragma Reduction of the array. However, due to the configuration on Xunil, the version of OpenMP does not support reduction on arrays. Therefore we had to impliment something similar to what reduction does for an array. 

In order to improve the speed of the histogram in parallel, we split the elements into multiple threads locally and then joined them together at the end. This showed an improvement of over 4x for the 100 million elements and 16 threads test run. 

## Code
The initialization of the histogram is parallelized using a pragma since there aren't any overlapping memory spaces where there needs to be serialization.
```C
   // Initialize histogram
   #pragma omp parallel shared(histogram, input_data) private(i)
   {
   #pragma omp for
   for(i = 0; i < histogram_size; i++)
      histogram[i] = 0;
   }
```

A normal pragma with parallel would render the incorrect histogram and sometimes would show significantly worse results. In order to overcome this, reduction on the array was needed. First, memory was allocated for each of the threads using a malloc. For each thread, the histogram was split so the input_data could be processed in parallel without any memory overlaps. A Pragma Omp Critical was used to ensure all the input_data was processed and threads completed before movign on. In the last step all of the private histograms were put together. 

```C
#pragma omp parallel for shared(histogram, input_data, histogram_size, num_elements)   private(n, i, priv_histo)
   for(n = 0; n < num_threads; n++)
   {
     priv_histo = (int*) malloc(sizeof(int)*histogram_size);

     for(i = 0; i < histogram_size; i++)
       priv_histo[i] = 0;

     for(i = omp_get_thread_num()*seg; i < num_elements - (num_threads - omp_get_thread_num() - 1)*seg; i++)
       priv_histo[input_data[i]]++;

     #pragma openmp critical
     {
       for (i = 0; i < histogram_size; i++)
       histogram[i] += priv_histo[i];
     }

   }
```




## Data

| Number of Elements |	Number of Threads	| Serial Time (s)	| Parallel Time (s) |	Times faster |
| ------------------|-------------|-----------------|-------------------|--------------|
| 1000000	| 2 |	0.0058	| 0.0073 | 0.7945 |
| 1000000	| 4 | 0.0049	| 0.0040 | 1.225 |
| 1000000	| 8 | 0.0054 | 0.0034 | 1.5882 |
| 1000000	| 16 | 0.0044 |	0.0031 | 1.4193 |
| 10000000	| 2 | 0.0446 | 0.0578 | 0.7716 |
| 10000000	| 4 | 0.0374 | 0.0304	| 1.230 |
| 10000000	| 8 | 0.0373 | 0.0167 | 2.2335 |
| 10000000	| 16 | 0.0449 |	0.0130 | 3.8385 |
| 100000000	| 2 | 0.4633 | 0.4506 | 1.0281 |
| 100000000	| 4 | 0.3695 | 0.2310 | 1.5996 |
| 100000000	| 8 | 0.4474 | 0.1285 | 3.4817 |
| 100000000	| 16 | 0.3754 |	0.1075	| 3.4921 |



# Graph for 100,000,000 element Histogram

![Image of Timing Graph for 100 Million Elements]
(https://github.com/om23/ecec413/blob/master/Assignment2/timingGraph.png)





### Extra Timing data
| Number of Elements |	Number of Threads	| Serial Time (s)	| Parallel Time (s) |
| ------------------|-------------|-----------------|-------------------|
| 1000000 | 2 | 0.0058 | 0.0077 |
| 1000000 | 4 | 0.0046 | 0.0039 |
| 1000000 | 8 | 0.0054 | 0.0026 |
| 1000000 | 16 | 0.0053 | 0.0025 |
| 10000000 | 2 | 0.0458 | 0.0565 |
| 10000000 | 4 | 0.0364 | 0.0300 |
| 10000000 | 8 | 0.0364 | 0.0147 |
| 10000000 | 16 | 0.0439 | 0.0141 |
| 100000000 | 2 | 0.3634 | 0.4132 |
| 100000000 | 4 | 0.4433 | 0.2319 |
| 100000000 | 8 | 0.4412 | 0.1071 |
| 100000000 | 16 | 0.3643 | 0.1029 |
