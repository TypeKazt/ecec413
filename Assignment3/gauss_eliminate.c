/* Gaussian elimination code.
 * Author: Naga Kandasamy
 * Date created: 02/07/2014
 * Date of last update: 01/30/2017
 * Compile as follows: gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c -lpthread -std=c99 -lm
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <semaphore.h>
#include "gauss_eliminate.h"

#define MIN_NUMBER 2
#define MAX_NUMBER 50
#define num_threads 2
#define num_elements MATRIX_SIZE


/* Function prototypes. */
extern int compute_gold (float *, unsigned int);
Matrix allocate_matrix (int num_rows, int num_columns, int init);
void gauss_eliminate_using_pthreads (float *);
void rowReduction (void *);
void eliminationStep (void *);
int perform_simple_check (const Matrix);
void print_matrix (const Matrix);
float get_random_number (int, int);
int check_results (float *, float *, unsigned int, float);

struct s1 {
	int id;
	float* mat;
};

typedef struct barrier_struct_tag{
    sem_t counter_sem; /* Protects access to the counter. */
    sem_t barrier_sem; /* Signals that barrier is safe to cross. */
    int counter; /* The value itself. */
    int num_calls; 
} barrier_t;

void* reduceRow(void *s);
void* pthread_wrapper(void *s);
void barrier_sync(barrier_t *);


int row_number = 0;
int row_start = 0;
barrier_t p_barrier;
barrier_t p2_barrier;
pthread_mutex_t* row_mutexs;

int
main (int argc, char **argv)
{
  /* Check command line arguments. */
  if (argc > 1)
    {
      printf ("Error. This program accepts no arguments. \n");
      exit (0);
    }

  /* Matrices for the program. */
  Matrix A;			// The input matrix
  Matrix U_reference;		// The upper triangular matrix computed by the reference code
  Matrix U_mt;			// The upper triangular matrix computed by the pthread code

  /* Initialize the random number generator with a seed value. */
  srand (time (NULL));

  /* Allocate memory and initialize the matrices. */
  A = allocate_matrix (MATRIX_SIZE, MATRIX_SIZE, 1);	// Allocate and populate a random square matrix
  U_reference = allocate_matrix (MATRIX_SIZE, MATRIX_SIZE, 0);	// Allocate space for the reference result
  U_mt = allocate_matrix (MATRIX_SIZE, MATRIX_SIZE, 0);	// Allocate space for the multi-threaded result

  /* Copy the contents of the A matrix into the U matrices. */
  for (int i = 0; i < A.num_rows; i++)
    {
      for (int j = 0; j < A.num_rows; j++)
	{
	  U_reference.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
	  U_mt.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
	}
    }

  printf ("Performing gaussian elimination using the reference code. \n");
  struct timeval start, stop;
  gettimeofday (&start, NULL);
  int status = compute_gold (U_reference.elements, A.num_rows);
  gettimeofday (&stop, NULL);
  printf ("CPU run time = %0.4f s. \n",
	  (float) (stop.tv_sec - start.tv_sec +
		   (stop.tv_usec - start.tv_usec) / (float) 1000000));

  if (status == 0)
    {
      printf("Failed to convert given matrix to upper triangular. Try again. Exiting. \n");
      exit (0);
    }
  status = perform_simple_check (U_reference);	// Check that the principal diagonal elements are 1 
  if (status == 0)
    {
      printf ("The upper triangular matrix is incorrect. Exiting. \n");
      exit (0);
    }
  printf ("Single-threaded Gaussian elimination was successful. \n");

  gettimeofday (&start, NULL);
  /* Perform the Gaussian elimination using pthreads. The resulting upper triangular matrix should be returned in U_mt */
  gauss_eliminate_using_pthreads (U_mt.elements);
  gettimeofday (&stop, NULL);
  printf ("PThreads CPU run time = %0.4f s. \n",
  (float) (stop.tv_sec - start.tv_sec +
     (stop.tv_usec - start.tv_usec) / (float) 1000000));


  /* check if the pthread result is equivalent to the expected solution within a specified tolerance. */
  int size = MATRIX_SIZE * MATRIX_SIZE;
  int res = check_results (U_reference.elements, U_mt.elements, size, 0.001f);
  printf ("Test %s\n", (1 == res) ? "PASSED" : "FAILED");

  printf("row: %d\n", row_number);

  /* Free memory allocated for the matrices. */
  free (A.elements);
  free (U_reference.elements);
  free (U_mt.elements);

  return 0;
}


/* Write code to perform gaussian elimination using pthreads. */
void gauss_eliminate_using_pthreads (float *U_mt)
{

 //  if(pthread_barrier_init(&redux_barrier, NULL, num_threads) || 
	//  pthread_barrier_init(&elim_barrier, NULL, num_threads+1))
 //  {
 //    printf("Barrier creation failed\n");
	// exit(1);
 //  }

  /* Initialize the barrier data structure. */
  p_barrier.counter = 0;
  p_barrier.num_calls = num_threads + 1;
  sem_init(&p_barrier.counter_sem, 0, 0); /* Initialize the semaphore protecting the counter to unlocked. */
  sem_init(&p_barrier.barrier_sem, 0, 0); /* Initialize the semaphore protecting the barrier to locked. */

  p2_barrier.counter = 0;
  p2_barrier.num_calls = num_threads;
  sem_init(&p2_barrier.counter_sem, 0, 0); /* Initialize the semaphore protecting the counter to unlocked. */
  sem_init(&p2_barrier.barrier_sem, 0, 0); /* Initialize the semaphore protecting the barrier to locked. */



  //create threads
  pthread_t threads[num_threads];
  int i, j, n, m;
  unsigned int elements;
  struct s1* para = (struct s1*) malloc(num_threads * sizeof(struct s1));
  for (i = 0; i < num_threads; i++)
  {
    para[i].mat = U_mt;
    para[i].id = i;
    // creating num_threads pthreads
    pthread_create(&threads[i], NULL, pthread_wrapper, (void *)&para[i]);
  }

  //main thread row loop
  for(elements=0; elements < num_elements; elements++)
  {
    row_number = elements;
    row_start = 1;
    barrier_sync(&p_barrier);
    U_mt[num_elements * elements + elements] = 1; //TODO fix, needs to be done only once
    //printf("main row: %d \n", row_number);
    barrier_sync(&p_barrier);
  }
}


void* pthread_wrapper(void *s)
{
  struct s1* myStruct = (struct s1*) s;
	while(row_number < MATRIX_SIZE)
	{
		while(!row_start){}
		barrier_sync(&p2_barrier);
		row_start = 0; //TODO fix, needs to only be done once
    //printf("thread row: %d \n", row_number);
		rowReduction(s);
		barrier_sync(&p_barrier);
    eliminationStep(s);
		barrier_sync(&p_barrier); 
	}
	pthread_exit(0);
}

void rowReduction(void *s) {
  int p;
  struct s1* myStruct = (struct s1*) s;
  int elements = row_number;
  int id = myStruct->id;
  float* U_mt = myStruct->mat;
  printf("NUM_THREADS: %d, %d \n", num_threads, num_elements);
  printf("Elements: %d\n", elements);
  for (p = elements+id+1; p < num_elements;)
  {
    U_mt[num_elements * elements + p] = (float) (U_mt[num_elements * elements + p] / U_mt[num_elements * elements + elements]); // division step
    printf("U_mt element: %f \n", U_mt[num_elements * elements + p]);
    printf("p: %d, thread_id %d, \n", p, id);
    p += num_threads;
  }
  //U_mt[num_elements * elements + elements] = 1; //TODO fix, needs to be done only once
}


void eliminationStep(void *s) {
  int  b, c;
  struct s1* myStruct = (struct s1*) s;
  int elements = row_number;
  int id = myStruct->id;
  float* U_mt = myStruct->mat;

  for (b = (elements + id)+1; b < num_elements; b += num_threads)
  {
    for (c = elements+1; c < num_elements; c++)
    {
      U_mt[num_elements * b + c] = U_mt[num_elements * b + c] - (U_mt[num_elements * b + elements] * U_mt[num_elements * elements + c]); // elimination step
    }
    U_mt[num_elements * b + elements] = 0;
  }

}


/* Function checks if the results generated by the single threaded and multi threaded versions match. */
int
check_results (float *A, float *B, unsigned int size, float tolerance)
{
  for (int i = 0; i < size; i++)
    if (fabsf (A[i] - B[i]) > tolerance)
	{
	  printf("failed elm: %d\n", i);
	  printf("A[i]: %f  B[i]: %f\n", A[i], B[i]);
      return 0;
	}
  return 1;
}


/* Allocate a matrix of dimensions height*width
 * If init == 0, initialize to all zeroes.  
 * If init == 1, perform random initialization. 
*/
Matrix
allocate_matrix (int num_rows, int num_columns, int init)
{
  Matrix M;
  M.num_columns = M.pitch = num_columns;
  M.num_rows = num_rows;
  int size = M.num_rows * M.num_columns;

  M.elements = (float *) malloc (size * sizeof (float));
  for (unsigned int i = 0; i < size; i++)
    {
      if (init == 0)
	M.elements[i] = 0;
      else
	M.elements[i] = get_random_number (MIN_NUMBER, MAX_NUMBER);
    }
  return M;
}


/* Returns a random floating-point number between the specified min and max values. */ 
float
get_random_number (int min, int max)
{
  return (float)
    floor ((double)
	   (min + (max - min + 1) * ((float) rand () / (float) RAND_MAX)));
}

/* Performs a simple check on the upper triangular matrix. Checks to see if the principal diagonal elements are 1. */
int
perform_simple_check (const Matrix M)
{
  for (unsigned int i = 0; i < M.num_rows; i++)
    if ((fabs (M.elements[M.num_rows * i + i] - 1.0)) > 0.001)
      return 0;
  return 1;
}


/* The function that implements the barrier synchronization. */
void 
barrier_sync(barrier_t *barrier)
{
    sem_wait(&(barrier->counter_sem));

    /* Check if all threads before us, that is NUM_THREADS-1 threads have reached this point. */    
    if(barrier->counter == (barrier->num_calls - 1)){
        barrier->counter = 0; /* Reset the counter. */
        sem_post(&(barrier->counter_sem)); 
           
        /* Signal the blocked threads that it is now safe to cross the barrier. */
        //printf("Thread number %d is signalling other threads to proceed. \n", thread_number); 
        for(int i = 0; i < (barrier->num_calls - 1); i++)
            sem_post(&(barrier->barrier_sem));
    } 
    else{
        barrier->counter++;
        sem_post(&(barrier->counter_sem));
        sem_wait(&(barrier->barrier_sem)); /* Block on the barrier semaphore. */
    }
}
