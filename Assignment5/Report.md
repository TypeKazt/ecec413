# Parallel Computing Assignment #5
### By: Alex Kazantsev and Hanumant Mahida
### Date: February 28th, 2017

## Results
Using SSE, the Gaussian Elimination was futher optimized. Our results on Xunil-03 achieved results of 1.64x faster for a 2048 by 2048 matrix. 

## Code
Set SSE Instructions
```C
  float *ptr;
  __m128 *src;
  __m128 op_1;
  __m128 op_2;
```

Parts of the U matrix are set and allocated and then a division step is run using SSE.
```C
  src = (__m128 *)(U.elements + (num_elements * k + j));
  op_1 = _mm_set_ps1(U.elements[num_elements * k + k]);
  for (; j < num_elements; j += 4)
  {
      *src = _mm_div_ps(*src, op_1);
      ++src;
  }
```

SSE is used for the substitution step and the laod, multiply, and subtraction instructions are utilized.
```C
  src = (__m128 *)(U.elements + (num_elements * i + j));
  op_1 = _mm_set_ps1(U.elements[num_elements * i + k]);
  for (; j < num_elements; j += 4)
  {
      op_2 = _mm_load_ps(U.elements + (num_elements * k) + j);
      op_2 = _mm_mul_ps(op_1, op_2);
      *src = _mm_sub_ps(*src, op_2);
      ++src;
  }
```
  
### Timing data
| Number of Elements | Serial Time (s)	| SSE Time (s) | Times faster |
|-------------|-----------------|-------------------|--------------|
| 2048	| 4.72 | 2.87 | 1.64 |
