# Report of Applying OpenMP to Histograms

### By: Alex Kazantsev and Hanumant Mahida
### Date: February 2th, 2017

## Implimentation & Results


## Code
```C
```


```C
```




## Data

| Number of Elements |	Number of Threads	| Serial Time (s)	| Parallel Time (s) |	Times faster |
| ------------------|-------------|-----------------|-------------------|--------------|
| 100000000	| 2 |	0.0058	| 0.0073 | 0.7945 |
| 100000000	| 4 | 0.0049	| 0.0040 | 1.225 |
| 100000000	| 8 | 0.0054 | 0.0034 | 1.5882 |
| 100000000	| 16 | 0.0044 |	0.0031 | 1.4193 |



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
