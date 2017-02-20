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

#define num_threads 16
#define num_elements MATRIX_SIZE



/////////////////////////////////////////////////////////////////////////////
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
	chol_using_pthreads(A, U_pthreads);

	/* MODIFY THIS CODE: Perform the Cholesky decomposition using openmp. The resulting upper traingular matrix should be returned in U_openmp */
	printf("executing openmp\n");
	gettimeofday(&start, NULL);
	chol_using_openmp(A, U_openmp);
	gettimeofday(&stop, NULL);
	printf("CPU run time openmp = %0.2f s. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));


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

	//print_matrix(U_openmp);

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
	
}

/* Write code to perform Cholesky decopmposition using openmp. */
void chol_using_openmp(const Matrix A, Matrix U)
{

	unsigned int i, j, k;  
    unsigned int size = A.num_rows * A.num_columns;
	omp_set_num_threads(num_threads);
	int num_rows = num_rows;

    // Perform the Cholesky decomposition in place on the U matrix
			for(k = 0; k < U.num_rows; k++){
					  // Take the square root of the diagonal element
					  	U.elements[k * U.num_rows + k] = sqrt(U.elements[k * U.num_rows + k]);

					  // Division step
					  for(j = (k + 1); j < U.num_rows; j++){
								 U.elements[k * U.num_rows + j] /= U.elements[k * U.num_rows + k]; // Division step
					  }

					  // Elimination step
					 
					  #pragma openmp parallel for schedule(static) private(i,j)
					  for(i = (k + 1); i < U.num_rows; i++){
								 for(j = i; j < U.num_rows; j++)
											U.elements[i * U.num_rows + j] -= U.elements[k * U.num_rows + i] * U.elements[k * U.num_rows + j]; 
					}
			}   
	  
    // As the final step, zero out the lower triangular portion of U
    for(i = 0; i < U.num_rows; i++)
              for(j = 0; j < i; j++)
                         U.elements[i * U.num_rows + j] = 0.0;

    return 1;
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




