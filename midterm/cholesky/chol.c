/* Cholesky decomposition.
 * Compile as follows:
 * 						gcc -fopenmp -o chol chol.c chol_gold.c -lpthread -lm -std=c99
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "chol.h"
#include <pthread.h>
#include <semaphore.h>



////////////////////////////////////////////////////////////////////////////////
// declarations, forward

Matrix allocate_matrix(int num_rows, int num_columns, int init);
int perform_simple_check(const Matrix M);
void print_matrix(const Matrix M);
extern Matrix create_positive_definite_matrix(unsigned int, unsigned int);
extern int chol_gold(const Matrix, Matrix);
extern int check_chol(const Matrix, const Matrix);
int easy_check(const Matrix, const Matrix);
void chol_using_pthreads(const Matrix, Matrix);
void chol_using_openmp(const Matrix, Matrix);
void* pthread_wrapper(void*);
void rowReduction(void *s);
void eliminationStep(void* s);
void setZeroes(void* s);
//void barrier_sync(barrier_t *, int);


#define num_threads 16
#define num_elements MATRIX_SIZE

unsigned int row_number = 0;
//barrier_t barrier;

struct s1 {
    int id;
    float* mat;
};


pthread_barrier_t barrier_main;
pthread_barrier_t barrier_threads;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) 
{	
	// Check command line arguments
	if(argc > 1){
		printf("Error. This program accepts no arguments. \n");
		exit(0);
	}		
	 
	// Matrices for the program
	Matrix A; // The N x N input matrix
	Matrix reference; // The upper triangular matrix computed by the CPU
	Matrix U_pthreads; // The upper triangular matrix computed by the pthread implementation
	Matrix U_openmp; // The upper triangular matrix computed by the openmp implementation 
	
	// Initialize the random number generator with a seed value 
	srand(time(NULL));
	struct timeval start, stop; 
	// Create the positive definite matrix. May require a few tries if we are unlucky
	int success = 0;
	while(!success){
		A = create_positive_definite_matrix(MATRIX_SIZE, MATRIX_SIZE);
		if(A.elements != NULL)
				  success = 1;
	}
	// print_matrix(A);
	// getchar();


	reference  = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0); // Create a matrix to store the CPU result
	U_pthreads =  allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0); // Create a matrix to store the pthread result
	U_openmp =  allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0); // Create a matrix to store the openmp result
	for(int i = 0; i < num_elements*num_elements; i ++) 
	{
        U_pthreads.elements[i] = A.elements[i];
        reference.elements[i] = A.elements[i];
        U_openmp.elements[i] = A.elements[i];
	}
	// compute the Cholesky decomposition on the CPU; single threaded version	
	printf("Performing Cholesky decomposition on the CPU using the single-threaded version. \n");
	gettimeofday(&start, NULL);
	int status = chol_gold(A, reference);
	gettimeofday(&stop, NULL);
	printf("CPU run time single thread = %0.2f s. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
	if(status == 0){
			  printf("Cholesky decomposition failed. The input matrix is not positive definite. \n");
			  exit(0);
	}
	
	
	/*
 	 printf("Double checking for correctness by recovering the original matrix. \n");
	if(check_chol(A, reference) == 0){
		printf("Error performing Cholesky decomposition on the CPU. Try again. Exiting. \n");
		exit(0);
	}
	printf("Cholesky decomposition on the CPU was successful. \n");
*/

	/* MODIFY THIS CODE: Perform the Cholesky decomposition using pthreads. The resulting upper triangular matrix should be returned in 
	 U_pthreads */

	/* MODIFY THIS CODE: Perform the Cholesky decomposition using openmp. The resulting upper traingular matrix should be returned in U_openmp */
	printf("executing openmp\n");
	gettimeofday(&start, NULL);
	chol_using_openmp(A, U_openmp);
	gettimeofday(&stop, NULL);
	printf("CPU run time openmp = %0.2f s. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));

	printf("executing pthreads\n");
	gettimeofday(&start, NULL);
	chol_using_pthreads(A, U_pthreads);
	gettimeofday(&stop, NULL);
	printf("CPU run time pthreads = %0.2f s. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));

	// Check if the pthread and openmp results are equivalent to the expected solution
	/*
	if(check_chol(A, U_pthreads) == 0) 
			  printf("Error performing Cholesky decomposition using pthreads. \n");
	else
			  printf("Cholesky decomposition using pthreads was successful. \n");

	if(check_chol(A, U_openmp) == 0) 
			  printf("Error performing Cholesky decomposition using openmp. \n");
	else	
			  printf("Cholesky decomposition using openmp was successful. \n");
	*/	

	if(easy_check(reference, U_openmp) == 0) 
			  printf("Error performing Cholesky decomposition using openmp. \n");
	else	
			  printf("Cholesky decomposition using openmp was successful. \n");
	



	if(easy_check(reference, U_pthreads) == 0) 
			  printf("Error performing Cholesky decomposition using pthreads. \n");
	else	
			  printf("Cholesky decomposition using pthreads was successful. \n");
	
	// Free host matrices
	free(A.elements); 	
	free(U_pthreads.elements);	
	free(U_openmp.elements);
	free(reference.elements); 
	return 1;
}

/* Write code to perform Cholesky decopmposition using pthreads. */
void chol_using_pthreads(const Matrix A, Matrix U)
{
	if(pthread_barrier_init(&barrier_main, NULL, num_threads+1) ||
		pthread_barrier_init(&barrier_threads, NULL, num_threads))
 	{
    	printf("Barrier creation failed\n");
    	exit(1);
 	}

  /* Initialize the barrier data structure. */

     // create threads
  pthread_t threads[num_threads];
  int i, j, n, m;
  unsigned int row;
  struct s1* para = (struct s1*) malloc(num_threads * sizeof(struct s1));
  for (i = 0; i < num_threads; i++)
  {
    para[i].mat = U.elements;
    para[i].id = i;
    // creating num_threads pthreads
    pthread_create(&threads[i], NULL, pthread_wrapper, (void *)&para[i]);
  }

  //main thread row loop
  for(row = 0; row< num_elements; row++)
  {
    row_number = row;
	U.elements[row * U.num_rows + row] = sqrt(U.elements[row * U.num_rows + row]);
    pthread_barrier_wait(&barrier_main);
    pthread_barrier_wait(&barrier_main);
  }
}


void* pthread_wrapper(void* s)
{
	struct s1* myStruct = (struct s1*) s;
    while(row_number < MATRIX_SIZE)
    {
        pthread_barrier_wait(&barrier_main);
        rowReduction(s);
        pthread_barrier_wait(&barrier_threads);
        eliminationStep(s);
        pthread_barrier_wait(&barrier_threads);
		if (row_number == MATRIX_SIZE-1)
			setZeroes(s);
			
        pthread_barrier_wait(&barrier_main);
    }
    pthread_exit(0);
}
void rowReduction(void *s) {
  int p;
  struct s1* myStruct = (struct s1*) s;
  int elements = row_number;
  int id = myStruct->id;
  float* U_mt = myStruct->mat;
  for (p = elements+id+1; p < num_elements;)
  {
    U_mt[num_elements * elements + p] = (float) (U_mt[num_elements * elements + p] / U_mt[num_elements * elements + elements]); // division step
    p += num_threads;
  }
}


void eliminationStep(void *s) {
  int  b, c;
  struct s1* myStruct = (struct s1*) s;
  int elements = row_number;
  int id = myStruct->id;
  float* U_mt = myStruct->mat;

  for (b = (elements + id)+1; b < num_elements; )
  {
    for (c = elements+1; c < num_elements; c++)
    {
      U_mt[num_elements * b + c] = U_mt[num_elements * b + c] - (U_mt[num_elements * elements + b] * U_mt[num_elements * elements + c]); // elimination step
    }
    //U_mt[num_elements * b + elements] = 0;
    //printf("b: %d, num_elements: %d \n", b, num_elements);
    b += num_threads;
  }

}

void setZeroes(void *s)
{
	int row, elm;
	struct s1* mys = (struct s1*) s;
	for (row = mys->id; row < num_elements; row += num_threads)
	{
		for(elm = 0; elm < row; elm++)
			mys->mat[row * num_elements + elm] = 0.0;	
	}
}


/* Write code to perform Cholesky decopmposition using openmp. */
void chol_using_openmp(const Matrix A, Matrix U)
{

	unsigned int i, j, k;  
    unsigned int size = A.num_rows * A.num_columns;

	omp_set_num_threads(num_threads);

    // Copy the contents of the A matrix into the working matrix U
	//#pragma openmp parallel for shared(A, U) private(i)
    #pragma omp parallel for
    for (i = 0; i < size; i ++) 
        U.elements[i] = A.elements[i];

	//#pragma openmp barrier

    // Perform the Cholesky decomposition in place on the U matrix
	for(k = 0; k < U.num_rows; k++){
		// Take the square root of the diagonal element
		U.elements[k * U.num_rows + k] = sqrt(U.elements[k * U.num_rows + k]);

		if(U.elements[k * U.num_rows + k] <= 0){
			printf("Cholesky decomposition failed. \n");
			//return 0;
		}

		// Division step
		for(j = (k + 1); j < U.num_rows; j ++)
				 U.elements[k * U.num_rows + j] /= U.elements[k * U.num_rows + k];

		// Elimination step	
		#pragma omp parallel for private(j)
		for(i = (k + 1); i < U.num_rows; i++)
			for(j = i; j < U.num_rows; j++)
				U.elements[i * U.num_rows + j] -= U.elements[k * U.num_rows + i] * U.elements[k * U.num_rows + j]; 
		}   
	
    // As the final step, zero out the lower triangular portion of U
	//#pragma openmp parallel for private(i, j) shared(U)
    for(i = 0; i < U.num_rows; i++)
    	for(j = 0; j < i; j++)
              	U.elements[i * U.num_rows + j] = 0.0;

    // printf("The Upper triangular matrix is: \n");
    // print_matrix(U);

}


// Allocate a matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
Matrix allocate_matrix(int num_rows, int num_columns, int init)
{
    	Matrix M;
    	M.num_columns = M.pitch = num_columns;
    	M.num_rows = num_rows;
    	int size = M.num_rows * M.num_columns;
		
	M.elements = (float *) malloc(size * sizeof(float));
	for(unsigned int i = 0; i < size; i++){
		if(init == 0) M.elements[i] = 0; 
		else
			M.elements[i] = (float)rand()/(float)RAND_MAX;
	}
    return M;
}	

