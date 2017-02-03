# Report of Applying OpenMP to Histograms

### By: Alex Kazantsev and Hanumant Mahida
### Date: Febuary 2th, 2017

## Implimentation & Results

## Code
```C

```


```C
```


```C
```




## Data

| Number of Elements |	Number of Threads	| Serial Time (s)	| Parallel Time (s) |	Times faster |
| ------------------|-------------|-----------------|-------------------|--------------|
| 1000000	| 2 |	0.0058	| 0.0073 | 0.7945 |
| 1000000	| 4 | 0.0049	| 0.0040 | 1.225 |
| 1000000	| 8 | 0.0054 | 0.0034 | 1.5882 |
| 1000000	| 16 | 0.0044 |	0.0031 | 1.4193 |
| 10000000	| 2 | 0.0446 | 0.0578 | 0.7716 |
| 10000000	| 4 | 0.0374 | 0.0304	| 0.1230 |
| 10000000	| 8 | 0.0373 | 0.0167 | 2.2335 |
| 10000000	| 16 | 0.0449 |	0.0130 | 3.8385 |
| 100000000	| 2 | 0.4633 | 0.4506 | 1.0281 |
| 100000000	| 4 | 0.3695 | 0.2310 | 1.5996 |
| 100000000	| 8 | 0.4474 | 0.1285 | 3.4817 |
| 100000000	| 16 | 0.3754 |	0.1075	| 4.4921 |



# Graph for 100,000,000 element Histogram

![Image of Timing Graph for 100 Million Elements]
(https://github.com/om23/ecec413/blob/master/Assignment2/timingGraph.png)

