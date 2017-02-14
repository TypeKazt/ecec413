# Report of Applying OpenMP to Histograms

### By: Alex Kazantsev and Hanumant Mahida
### Date: February 12th, 2017

## Implimentation & Results


## Code
Struct used to allow each thread access to data for integral calculation.
```C
struct s1 {
	int id;
	double integral;
	float a;
	int n;
	float h;
};
```

Creating pthreads and assigning data to each pthread while calling the trapCalc function.
```C
for (i = 0; i < NUM_THREADS; i++)
	{
		para[i].id = i;
		para[i].integral = 0; // note: set as 0 since it would be redundant to calculate first integral for each thread 
   		para[i].a = a;
   		para[i].n = n;
   		para[i].h = h;
   		pthread_create(&threads[i], NULL, trapCalc, (void *)&para[i]);
	}
```

Joining all the pthreads together after trapCalc is finished.
```C
for (j = 0; j < NUM_THREADS; j++)
{
	pthread_join(threads[j], NULL);
}
```

Sums up integral values from each thread to get the total integral
```C
for (z = 0; z < NUM_THREADS; z++)
{
	integral += para[z].integral;
}
```

trapCalc function that each pthread uses to calculate the integral for their interval. Uses struct to its data and saves its integral calculation to the struct.
```C
void *trapCalc(void *s) 
{
	int k;
	struct s1* myStruct = (struct s1*) s;
	int id = myStruct->id;
	double integral = myStruct->integral;
	float a = myStruct->a;
	int n = myStruct->n;
	float h = myStruct->h;

	for (k = id; k <= n-1; k+=NUM_THREADS)
	{
		integral += f(a+k*h);
	}
	myStruct->integral = integral;

	pthread_exit(0);
}
```



## Data

| Number of Elements |	Number of Threads	| Serial Time (s)	| Parallel Time (s) |	Times faster |
| ------------------|-------------|-----------------|-------------------|--------------|
| 100000000	| 2 | 4.0540 | 2.1503 | 1.89 |
| 100000000	| 4 | 4.2359 | 1.0922 | 3.88 |
| 100000000	| 8 | 4.2469 | 0.6006 | 7.07 |
| 100000000	| 16 | 4.2387 |	0.3449 | 12.29 |



# Graph of Timing Results

![Image of Timing Graph for 100 Million Elements]
(https://github.com/om23/ecec413/blob/master/Assignment4/timingGraph.png)


