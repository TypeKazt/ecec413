# Parallel Computing Midterm
### By: Alex Kazantsev and Hanumant Mahida
### Date: February 19th, 2017

## Results

![Timing Graphs]
 (https://github.com/om23/ecec413/blob/master/midterm/)


## Code


### Problem 1


### Problem 2

#### Timing data from Pthreads without barriers (compiled using C99)
| Matrix Size | Number of Threads  | Serial Time (s)	| Red-Black (s) | Jacobi (s) | Serial vs Red-Black | Serial vs Jacobi |
| ------------------|-------------|-----------------|-------------------|--------------|------------|------------|
| 8192 | 4	| 75.36 | 11.73 | 10.89 | 6.42 | 6.92 |
| 8192 | 8	| 75.60 | 5.89 | 4.51 | 12.84 | 16.76 |
| 8192 | 16 | 75.86 | 2.79 | 2.16 | 27.19 | 35.12 |



#### Red-Black Decomposition

Call OpenMP parallel on while loop
```C
 #pragma omp parallel
	for (int i = 1; i < (grid_2->dimension-1); i++)
	{
		....
	}

```

Computing X (Red) for decomposition
```C
// Compute "X" (odd)	
	xStart = (id*2) + i%2;
	#pragma omp for reduction(+: diff_x) private(temp, j, xStart)
	for (int j = xStart; j < (grid_2->dimension-1); j+=num_threads)
	{
		temp = grid_2->element[i * grid_2->dimension + j];
		grid_2->element[i * grid_2->dimension + j] = 0.20*(grid_2->element[i * grid_2->dimension + j] + 
				grid_2->element[(i - 1) * grid_2->dimension + j] +
				grid_2->element[(i + 1) * grid_2->dimension + j] +
				grid_2->element[i * grid_2->dimension + (j + 1)] +
				grid_2->element[i * grid_2->dimension + (j - 1)]);
		diff_x = diff_x + fabs(grid_2->element[i * grid_2->dimension + j] - temp);	
	}
```



Computing Y (Black) for decomposition
```C
// Compute "Y" (even)
	yStart = id*2 + (1 - id%2);
	#pragma omp for reduction(+: diff_y) private(temp, j, yStart)
	for (int k = yStart; k < (grid_2->dimension-1); k+=num_threads)
	{
		temp = grid_2->element[i * grid_2->dimension + j];
		grid_2->element[i * grid_2->dimension + j] = 0.20*(grid_2->element[i * grid_2->dimension + j] + 
				grid_2->element[(i - 1) * grid_2->dimension + j] +
				grid_2->element[(i + 1) * grid_2->dimension + j] +
				grid_2->element[i * grid_2->dimension + (j + 1)] +
				grid_2->element[i * grid_2->dimension + (j - 1)]);
		diff_y = diff_y + fabs(grid_2->element[i * grid_2->dimension + j] - temp);	
	}
```

#### Element-Based Decomposition

Using Reduction to parallelize data and also using a copy of the grid
```C
	while(!done){
		diff = 0;
		#pragma omp parallel for reduction(+: temp) 
		for (int i = 1; i < (grid_3->dimension-1); i++)
		{
			int id = omp_get_thread_num();
			for (int j = id; j < (grid_3->dimension-1); j+=num_threads)
			{
				temp = grid_3->element[i * grid_3->dimension + j];
				grid_3->element[i * grid_3->dimension + j] = 0.20*(grid_3_copy->element[i * grid_3_copy->dimension + j] + 
						grid_3_copy->element[(i - 1) * grid_3_copy->dimension + j] +
						grid_3_copy->element[(i + 1) * grid_3_copy->dimension + j] +
						grid_3_copy->element[i * grid_3_copy->dimension + (j + 1)] +
						grid_3_copy->element[i * grid_3_copy->dimension + (j - 1)]);
				diff = diff + fabs(grid_3->element[i * grid_3->dimension + j] - temp);	
			}
		}
```






