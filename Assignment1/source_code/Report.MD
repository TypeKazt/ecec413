# Report of Applying OpenMP to Gaussian Elimination

### By: Alex Kazantsev and Hanumant Mahida
### Date: January 29th, 2017

## Implimentation & Results
OpenMP was used to parallelize some of the functionality of calculating gaussian elimination. The time for a serial program is about 0.64 seconds on a 24-core Intel CPU for a 1024 by 1024 matrix. By utilizing OpenMP Pragmas, the time can be improved. Our results showed the paraellelized program taking an average of 0.45 seconds. This improvement was due to using two Pragmas and resulted in passing tests.

We did see better improvements, an average of .18 seconds, however the tests failed. After investigating, we found that for the U[k][k] element of the matrix, the value would not match against the test matrix. 


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
