# Parallel Computing Midterm
### By: Alex Kazantsev and Hanumant Mahida
### Date: February 19th, 2017

## Results

![Timing Graphs]
 (https://github.com/om23/ecec413/blob/master/midterm/solver.png)

![Timing Graphs]
 (https://github.com/om23/ecec413/blob/master/midterm/chol.png)


## Code


### Problem 1
Note: Compile using  gcc -fopenmp -o chol chol.c chol_gold.c -lpthread -lm -std=gnu99


#### Timing data
| Matrix Size | Number of Threads  | Serial Time (s) | OpenMP (s) | PThreads (s) | Serial vs OpenMP | Serial vs PThreads |
| ------------------|-------------|-----------------|-------------------|--------------|------------|------------|
| 512 | 2| 0.10 | 0.15 | 0.10 | 0.67 | 1 |
| 512 | 4| 0.09 | 0.10 | 0.09 | 0.9 | 1|
| 512 | 8 | 0.09 | 0.13 | 0.08 | 0.69 | 1.125| 
| 512 | 16 | 0.10 | 0.11 | 0.09 | 0.91 | 1.11|
| 1024 | 2 | 0.72 | 1.09 | 0.72 | 0.66 | 1 |
| 1024 | 4 | 0.69 | 0.70 | 0.50 | 0.98 | 1.38 |
| 1024 | 8 | 0.71 | 1.09 | 0.47 | 1.06 | 1.37 |
| 1024 | 16	| 0.70 | 0.66 | 0.51 | 1.06 | 1.37 |
| 2048 | 2 | 5.67 | 8.58 | 5.76 | 0.66 | 0.98 |
| 2048 | 4 | 5.59 | 5.39 | 4.02 | 1.04 | 1.39 |
| 2048 | 8 | 5.59 | 6.27 | 3.78  | 0.89 | 1.47 |
| 2048 | 16 | 5.56 | 4.22 | 3.61 | 1.31 | 1.54 |
 

### Problem 2
Note: Compile using  gcc -o solver solver.c solver_gold.c -fopenmp -std=c99 -lm -lpthread

#### Timing data
| Matrix Size | Number of Threads  | Serial Time (s) | Red-Black (s) | Jacobi (s) | Serial vs Red-Black | Serial vs Jacobi |
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






