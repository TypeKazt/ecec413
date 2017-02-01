# Report of Applying OpenMP to Gaussian Elimination

### By: Alex Kazantsev and Hanumant Mahida
### Date: January 29th, 2017

## Implimentation & Results
OpenMP was used to parallelize some of the functionality of calculating gaussian elimination. For example, the time for a serial program is about 257.74 seconds on 16 threads for a 8192 by 8192 matrix. By utilizing OpenMP Pragmas, the time can be improved. Our results showed the paraellelized program taking an average of 38.83 seconds which was a 5.6 times faster improvement. This was due to using two Pragmas and resulted in passing tests.

## Code
```C
#pragma omp parallel for shared(A, U) private(i, j, k)
    for (i = 0; i < num_elements; i++)             /* Copy the contents of the A matrix into the U matrix. */
        for(j = 0; j < num_elements; j++)
            U[num_elements * i + j] = A[num_elements*i + j];
	#pragma omp barrier
```

A Pragma was used during the copying part of the program allowing for OpenMP to parallelize this functionality. This resulted in a minor speed improvement.

```C
for (k = 0; k < num_elements; k++){             /* Perform Gaussian elimination in place on the U matrix. */
		#pragma omp parallel for 
        for (j = (k + 1); j < num_elements; j++){   /* Reduce the current row. */
			if (U[num_elements*k + k] == 0){
				printf("0 val: %f\n", U[num_elements*k + k]);
				printf("0 val: %f\n", A[num_elements*k + k]);
				printf("Numerical instability detected. The principal diagonal element is zero. \n");
			}
            /* Division step. */
			U[num_elements * k + j] = (float)(U[num_elements * k + j] / U[num_elements * k + k]);
		}
			//printf("vala: %f\n", U[num_elements*k + j]);
		#pragma omp barrier
```

A Pragma was used to parallelize the row reduction and division steps of the program. This showed a significant speed improvement.

```C
#pragma omp parallel for 
        for (i = (k+1); i < num_elements; i++){
            for (j = (k+1); j < num_elements; j++){
                /* Elimnation step. */
				U[num_elements * i + j] = U[num_elements * i + j] -\
                                          (U[num_elements * i + k] * U[num_elements * k + j]);
			//	printf("valb: %f\n", U[num_elements*i + j]);
			}
            U[num_elements * i + k] = 0; 
		}
```

The last Pragma used was for the elimination step of the program. This also showed a significant speed improvement. 



## Data

| Number of Threads |	Matrix Size	| Serial Time (s)	| Parallel Time (s) |	Times faster |
| ------------------|-------------|-----------------|-------------------|--------------|
| 2	| 1024 | 0.55 |	0.25	| 1.2 |
| 2	| 2048 | 3.84 | 1.96	| 0.959183673 |
| 2	| 4096 | 33.75 | 17.69 | 0.907857547 |
| 2	| 8192 |	380.72 |	195	| 0.952410256 |
| 4	| 1024 | 0.54 |	0.18 | 2 | 
| 4	| 2048 | 3.77	|1.11	| 2.396396396 |
| 4	| 4096 |	31.32	| 8.8	| 2.559090909 |
| 4	| 8192	| 382.18	| 108.4|	2.525645756 |
| 8	| 1024	| 0.54	| 0.15	| 2.6 |
| 8	| 2048 |	3.72 |	0.63 |	4.904761905 |
| 8	| 4096	| 31.51	| 9.34	| 2.37366167 |
| 8	| 8192	| 259.73	| 41.14	| 5.313320369 |
| 16	| 1024	| 0.54	| 0.2	| 1.7 |
| 16	| 2048	| 3.76	| 0.72	| 4.222222222 |
| 16	| 4096	| 31.88	| 4.78	| 5.669456067 |
| 16	| 8192	| 257.74	| 38.83	| 5.637651301 |
