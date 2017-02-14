/*  Purpose: Calculate definite integral using trapezoidal rule.
 *
 * Input:   a, b, n
 * Output:  Estimate of integral from a to b of f(x)
 *          using n trapezoids.
 *
 * Compile: gcc -o trap trap.c -lpthread -lm
 * Usage:   ./trap
 *
 * Note:    The function f(x) is hardwired.
 *
 */

#ifdef _WIN32
#  define NOMINMAX 
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>

#define LEFT_ENDPOINT 5
#define RIGHT_ENDPOINT 1000
#define NUM_TRAPEZOIDS 100000000
#define NUM_THREADS 4

void *trapCalc (void *);

struct s1 {
	int id;
	double integral;
	float a;
	int n;
	float h;
};

double compute_using_pthreads(float, float, int, float);
double compute_gold(float, float, int, float);

int main(void) 
{
	int n = NUM_TRAPEZOIDS;
	float a = LEFT_ENDPOINT;
	float b = RIGHT_ENDPOINT;
	float h = (b-a)/(float)n; // Height of each trapezoid  
	printf("The height of the trapezoid is %f \n", h);

	struct timeval start, stop;
	gettimeofday (&start, NULL);
	double reference = compute_gold(a, b, n, h);
	gettimeofday (&stop, NULL);
	printf ("CPU run time = %0.4f s. \n",
		(float) (stop.tv_sec - start.tv_sec +
			(stop.tv_usec - start.tv_usec) / (float) 1000000));
	printf("Solution computed in Serial = %f \n", reference);


	/* Write this function to complete the trapezoidal on the GPU. */
	gettimeofday (&start, NULL);
	double pthread_result = compute_using_pthreads(a, b, n, h);
	gettimeofday (&stop, NULL);
	printf ("PThreads CPU run time = %0.4f s. \n",
		(float) (stop.tv_sec - start.tv_sec +
			(stop.tv_usec - start.tv_usec) / (float) 1000000));
	printf("Solution computed using pthreads = %f \n", pthread_result);
} 


/*------------------------------------------------------------------
 * Function:    f
 * Purpose:     Compute value of function to be integrated
 * Input args:  x
 * Output: (x+1)/sqrt(x*x + x + 1)

 */
float f(float x) {
		  return (x + 1)/sqrt(x*x + x + 1);
}  /* f */

/*------------------------------------------------------------------
 * Function:    Trap
 * Purpose:     Estimate integral from a to b of f using trap rule and
 *              n trapezoids
 * Input args:  a, b, n, h
 * Return val:  Estimate of the integral 
 */
double compute_gold(float a, float b, int n, float h) {
   double integral;
   int k;

   integral = (f(a) + f(b))/2.0;
   for (k = 1; k <= n-1; k++) {
     integral += f(a+k*h);
   }
   integral = integral*h;

   return integral;
}  

/* Complete this function to perform the trapezoidal rule on the GPU. */
double compute_using_pthreads(float a, float b, int n, float h)
{
	double integral;
	int i, j, z;

    integral = (f(a) + f(b))/2.0;

	pthread_t threads[NUM_THREADS];

	struct s1* para = malloc(NUM_THREADS * sizeof(struct s1));

	for (i = 0; i < NUM_THREADS; i++)
	{
		para[i].id = i;
		para[i].integral = 0; // note: set as 0 since it would be redundant to calculate first integral for each thread 
   		para[i].a = a;
   		para[i].n = n;
   		para[i].h = h;
   		pthread_create(&threads[i], NULL, trapCalc, (void *)&para[i]);
	}

	for (j = 0; j < NUM_THREADS; j++)
	{
		//printf("thread id: %i, integral values: %f \n", para[j].id, para[j].integral);
		pthread_join(threads[j], NULL);
	}

	// compute integral from struct para
	for (z = 0; z < NUM_THREADS; z++)
	{
		integral += para[z].integral;
	}

	integral = integral*h;

	return integral;
}


void *trapCalc(void *s) 
{
	int k;
	struct s1* myStruct = (struct s1*) s;
	int id = myStruct->id;
	double integral = myStruct->integral;
	float a = myStruct->a;
	int n = myStruct->n;
	float h = myStruct->h;

	for (k = 1; k <= n-1; k++)
	{
		integral += f(a+k*h);
	}
	myStruct->integral = integral;
	//printf("Integral value inside function: %f \n", integral);

	pthread_exit(0);
}


