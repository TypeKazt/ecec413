# Parallel Computing Assignment #3
### By: Alex Kazantsev and Hanumant Mahida
### Date: February 14th, 2017

NOTE: 2 versions exist. One with barriers that needs to be compiled with GNU99 and one without barriers that can be compiled with C99.

Barriers compile as follows: gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c -lpthread -std=gnu99 -lm

W/out barriers compile as follows: gcc -o gauss_without_barriers gauss_without_barriers.c compute_gold.c -lpthread -std=c99 -lm

## Results

![PThreads Barriers Timing Graph for 4096x4096 Matrix]
 (https://github.com/om23/ecec413/blob/master/Assignment3/barriersTiming.png)

 ![PThreads w/out Barriers Timing Graph for 4096x4096 Matrix]
 (https://github.com/om23/ecec413/blob/master/Assignment3/pthreadsTiming.png)

## Code

Struct to allocate variables and matrix pointers for the pthreads
```C
struct s1 {
  int elements;
  int id;
  float* mat;
};
```

Creating pthreads for rowReduction function while assigning elements, thread id, and the matrix as pointers to pthread_create()
```C
for (i = 0; i < num_threads; i++)
{
  para[i].elements = elements;
  para[i].id = i;
  para[i].mat = U_mt;
  pthread_create(&threads[i], NULL, rowReduction, (void *)&para[i]);
}
```

Joining pthreads after rowReduction to act as a barrier before eliminationStep is computed.
```C
for (j = 0; j < num_threads; j++)
{
  pthread_join(threads[j], NULL);
}
```

Set principal diagonal in U to be 1
```C
	U_mt[num_elements * elements + elements] = 1;
```

Creating pthreads for eliminationStep function while assigning elements, thread id, and the matrix as pointers to pthread_create()
```C
for (n = 0; n < num_threads; n++)
{
  para[n].elements = elements;
  para[n].id = n;
  para[n].mat = U_mt;
  pthread_create(&threads[n], NULL, eliminationStep, (void *)&para[n]);
}
```

Joining pthreads after eliminationStep and freeing para struct
```C
for (m = 0; m < num_threads; m++)
{
  pthread_join(threads[m], NULL);
}
free(para);
```

rowReduction computed using input struct pointer
```C
void *rowReduction(void *s) {
  int p;
  struct s1* myStruct = (struct s1*) s;
  int elements = myStruct->elements;
  int id = myStruct->id;
  float* U_mt = myStruct->mat;

  for (p = elements+id+1; p < num_elements;)
  {
    U_mt[num_elements * elements + p] = (float) (U_mt[num_elements * elements + p] / U_mt[num_elements * elements + elements]); // division step
    p += num_threads;
  }

  pthread_exit(0);
}
```

eliminationStep computed using input struct pointer
```C
void *eliminationStep(void *s) {
  int  b, c;
  struct s1* myStruct = (struct s1*) s;
  int elements = myStruct->elements;
  int id = myStruct->id;
  float* U_mt = myStruct->mat;

  for (b = (elements + id)+1; b < num_elements; )
  {
    for (c = elements+1; c < num_elements; c++)
    {
      U_mt[num_elements * b + c] = U_mt[num_elements * b + c] - (U_mt[num_elements * b + elements] * U_mt[num_elements * elements + c]); // elimination step
    }
    U_mt[num_elements * b + elements] = 0;
    //printf("b: %d, num_elements: %d \n", b, num_elements);
    b += num_threads;
  }

  pthread_exit(0);
}
```


### Timing data from Pthreads without barriers (compiled using C99)
| Number of Threads | Number of Elements | Serial Time (s)	| Parallel Time (s) | Times faster |
| ------------------|-------------|-----------------|-------------------|--------------|
| 1024 | 2	| 2.8015 | 1.5239 | 1.84 |
| 1024 | 8	| 2.7723 | 1.4899 | 1.86 |
| 1024 | 16 | 2.8189 | 1.4692 | 1.92 |
| 1024 | 32 | 2.5865 | 2.7859 | 0.93 |
| 2048 | 2	| 22.7194 | 11.5292 | 1.97 |
| 2048 | 8  | 22.6655 | 3.6530 | 6.20 |
| 2048 | 16 | 20.7298 | 4.5216 | 4.58 |
| 2048 | 32 | 21.4592 | 6.4329 | 3.34 |
| 4096 | 2  | 181.4115 | 89.6642 | 2.02 |
| 4096 | 8	| 175.1705 | 28.2202 | 6.21 |
| 4096 | 16 | 170.1095 | 23.0378 | 7.38 |
| 4096 | 32 | 169.4859 | 24.4565 | 6.93 |



### Timing data from Pthreads with barriers (compiled using GNU99)
| Number of Elements |	Number of Threads	| Serial Time (s)	| Parallel Time (s) | Speed Improvement |
| ------------------|-------------|-----------------|-------------------|--------------|
| 1024 | 4 | 3.6031 | 1.0670 | 3.38 |
| 1024 | 8 | 3.5868 | 0.9099 | 3.94 |
| 1024 | 16 | 3.3345 | 0.888 | 3.76 |
| 1024 | 32 | 3.5413 | 1.4444 | 2.45 |
| 2048 | 4 | 27.8267 | 6.2582 | 4.45 |
| 2048 | 8 | 27.751 | 3.7775 | 7.35 |
| 2048 | 16 | 27.5669 | 2.6006 | 10.60 |
| 2048 | 32 | 26.9078 | 5.1234 | 5.25 |
| 4096 | 4 | 222.0884 | 47.4154 | 4.68 |
| 4096 | 8 | 222.8902 | 27.8514 | 8.00 |
| 4096 | 16 | 215.8675 | 16.2313 | 13.30 |
| 4096 | 32 | 215.8881 | 18.2630 | 11.82 |



