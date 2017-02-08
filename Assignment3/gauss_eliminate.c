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
#include "gauss_eliminate.h"

#define MIN_NUMBER 2
#define MAX_NUMBER 50

/* Function prototypes. */
extern int compute_gold (float *, unsigned int);
Matrix allocate_matrix (int num_rows, int num_columns, int init);
void gauss_eliminate_using_pthreads (float *, unsigned int);
int perform_simple_check (const Matrix);
void print_matrix (const Matrix);
float get_random_number (int, int);
int check_results (float *, float *, unsigned int, float);

struct s1 {
int elements;
int id;
};

void* reduceRow(void *s);


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
  gauss_eliminate_using_pthreads (U_mt.elements,  A.num_rows);
  gettimeofday (&stop, NULL);
  printf ("PThreads CPU run time = %0.4f s. \n",
  (float) (stop.tv_sec - start.tv_sec +
     (stop.tv_usec - start.tv_usec) / (float) 1000000));


  /* check if the pthread result is equivalent to the expected solution within a specified tolerance. */
  int size = MATRIX_SIZE * MATRIX_SIZE;
  int res = check_results (U_reference.elements, U_mt.elements, size, 0.001f);
  printf ("Test %s\n", (1 == res) ? "PASSED" : "FAILED");

  /* Free memory allocated for the matrices. */
  free (A.elements);
  free (U_reference.elements);
  free (U_mt.elements);

  return 0;
}


/* Write code to perform gaussian elimination using pthreads. */
void
gauss_eliminate_using_pthreads (float *U, unsigned int num_elements)
{
  unsigned int elements;

  for (elements = 0; elements < num_elements; elements++); // perform Gaussian elimination
  {
    pthread_t threads[num_elements];
    int i, j, n, m;
    struct s1* para = malloc(num_threads * sizeof(struct s1));
    for (i = 0; i < num_threads; i++)
    {
      para[i].elements = elements;
      para[i].id = i;
      // creating num_threads pthreads
      pthread_create(&threads[i], NULL, rowReduction, (void*) &para[i]);
    }

    for (j = 0; j < num_threads; j++)
    {
      pthread_join(threads[j], NULL);
    }
    //free(para); //FIXME: shouldn't do this here since I need it again, right??

    // TODO: need a barrier or similar here

    U[num_elements * elements + elements] = 1; // set principal diagonal in U to be 1

    for (n = 0; n < num_threads; n++)
    {
      para[n].elements = elements;
      para[n].elements = n;
      // create num_threads pthreads
      pthread_create(&threads[n], NULL, eliminationStep, (void*) &para[n]);
    }

    for (m = 0; m < num_threads; m++)
    {
      pthread_join(threads[m], NULL);
    }
    free(para);


  }

}


void* rowReduction(void *s) {
  int p;

  for (p = elements + 1; p < num_elements; p++)
  {
    U[num_elements * elements + p] = (float) (U[num_elements * elements + p] / U[num_elements * elements + elements]); // division step
  }

  pthread_exit(0);
}


void* eliminationStep(void *s) {
  int  b, c;

  for (b = (elements + 1); b < num_elements; b++)
  {
    for (c = (elements + 1); b < num_elements; b++)
    {
      U[num_elements * b + c] = U[num_elements * b + c] - (U[num_elements * b + c] * U[num_elements * b + c]); // elimination step
    }
    U[num_elements * b + c] = 0;
  }

  pthread_exit;
}


/* Function checks if the results generated by the single threaded and multi threaded versions match. */
int
check_results (float *A, float *B, unsigned int size, float tolerance)
{
  for (int i = 0; i < size; i++)
    if (fabsf (A[i] - B[i]) > tolerance)
      return 0;
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
