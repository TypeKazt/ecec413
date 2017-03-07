/* Gaussian elimination code.
 * Author: Naga Kandasamy, 10/24/2015
 *
 * Compile as follows: 
 * gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c -std=c99 -O3 -lm
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include "gauss_eliminate.h"
#include <xmmintrin.h> 
#define MIN_NUMBER 2
#define MAX_NUMBER 50
#define num_elements MATRIX_SIZE

extern int compute_gold(float*, const float*, unsigned int);
Matrix allocate_matrix(int num_rows, int num_columns, int init);
void gauss_eliminate_using_sse(const Matrix, Matrix);
int perform_simple_check(const Matrix);
void print_matrix(const Matrix);
float get_random_number(int, int);
int check_results(float *, float *, unsigned int, float);


int 
main(int argc, char** argv) {
	if(argc > 1){
		printf("Error. This program accepts no arguments. \n");
		exit(0);
	}	

	/* Allocate and initialize the matrices. */
	Matrix  A;                                              /* The N x N input matrix. */
	Matrix  U;                                              /* The upper triangular matrix to be computed. */

	srand(time(NULL));

	A  = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 1);      /* Create a random N x N matrix. */
	U  = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0);      /* Create a random N x 1 vector. */

	/* Gaussian elimination using the reference code. */
	Matrix reference = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0);
	struct timeval start, stop;	
	gettimeofday(&start, NULL);

	printf("Performing gaussian elimination using the reference code. \n");
	int status = compute_gold(reference.elements, A.elements, A.num_rows);

	gettimeofday(&stop, NULL);
	printf("CPU run time = %0.2f s. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));

	if(status == 0){
		printf("Failed to convert given matrix to upper triangular. Try again. Exiting. \n");
		exit(0);
	}
	status = perform_simple_check(reference); // Check that the principal diagonal elements are 1 
	if(status == 0){
		printf("The upper triangular matrix is incorrect. Exiting. \n");
		exit(0); 
	}
	printf("Gaussian elimination using the reference code was successful. \n");

	/* WRITE THIS CODE: Perform the Gaussian elimination using the SSE version. 
	 * The resulting upper triangular matrix should be returned in U
	 * */
	gettimeofday(&start, NULL);
	gauss_eliminate_using_sse(A, U);
	gettimeofday(&stop, NULL);
	printf("SSE run time = %0.2f s! \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));

	/* Check if the SSE result is equivalent to the expected solution. */
	int size = MATRIX_SIZE*MATRIX_SIZE;
	int res = check_results(reference.elements, U.elements, size, 0.001f);
	printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");

	free(A.elements); A.elements = NULL;
	free(U.elements); U.elements = NULL;
	free(reference.elements); reference.elements = NULL;

	return 0;
}


	void 
gauss_eliminate_using_sse(const Matrix A, Matrix U)                  /* Write code to perform gaussian elimination using OpenMP. */
{
	unsigned int i, j, k;

	float *ptr;
	__m128 *src;
	__m128 op_1;
	__m128 op_2;

	for (i = 0; i < num_elements; i ++ )
	{
		for (j = 0; j < num_elements; j++)
		{
			U.elements[num_elements * i + j ] = A.elements[num_elements*i + j];
		}
	}



	for (k = 0; k < num_elements; k++)
	{
		for (j = (k + 1); j < num_elements; j++)
		{

			if(!((long)(U.elements + (num_elements * k) + j) & 0xF))
			{
				break;
			}	

			// division step
			U.elements[num_elements * k + j] = (float)(U.elements[num_elements * k + j] / U.elements[num_elements * k + k]);
		}

		src = (__m128 *)(U.elements + (num_elements * k + j));
		op_1 = _mm_set_ps1(U.elements[num_elements * k + k]);

		for (; j < num_elements; j += 4)
		{
			*src = _mm_div_ps(*src, op_1);
			++src;
		}

		if (j != num_elements)
		{
			j -= 4;
			for (; j < num_elements; ++j)
			{
				U.elements[num_elements * k + j] = (float)(U.elements[num_elements * k + j] / U.elements[num_elements * k + k]);
			}
		}

		U.elements[num_elements * k + k] = 1;  // set principal diagonal entry in U to be 1

		for (i = (k + 1); i < num_elements; i++)
		{
			for (j = (k + 1); j < num_elements; j++)
			{
				if (!((long)(U.elements + (num_elements * k) + j) & 0xF))
				{
					break;
				}
				U.elements[num_elements * i + j] = U.elements[num_elements * i + j] - (U.elements[num_elements * i + k] * U.elements[num_elements * k + j]);
			}

			src = (__m128 *)(U.elements + (num_elements * i + j));
			op_1 = _mm_set_ps1(U.elements[num_elements * i + k]);

			for (; j < num_elements; j += 4)
			{
				op_2 = _mm_load_ps(U.elements + (num_elements * k) + j);
				op_2 = _mm_mul_ps(op_1, op_2);
				*src = _mm_sub_ps(*src, op_2);
				++src;
			}

			if (j != num_elements)
			{
				j -= 4;
				for (; j < num_elements; j++)
				{
					U.elements[num_elements * i + j] = U.elements[num_elements * i + j] - (U.elements[num_elements * i + k] * U.elements[num_elements * k + j]);
				}
			}


			U.elements[num_elements * i + k] = 0;	
		}
	}



}


	int 
check_results(float *A, float *B, unsigned int size, float tolerance)   /* Check if refernce results match multi threaded results. */
{
	for(int i = 0; i < size; i++)
		if(fabsf(A[i] - B[i]) > tolerance)
			return 0;

	return 1;
}


/* Allocate a matrix of dimensions height*width. 
 * If init == 0, initialize to all zeroes.  
 * If init == 1, perform random initialization.
 * */
Matrix 
allocate_matrix(int num_rows, int num_columns, int init){
	Matrix M;
	M.num_columns = M.pitch = num_columns;
	M.num_rows = num_rows;
	int size = M.num_rows * M.num_columns;
	M.elements = (float*) malloc(size*sizeof(float));

	for(unsigned int i = 0; i < size; i++){
		if(init == 0) M.elements[i] = 0; 
		else
			M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	}

	return M;
}	


float 
get_random_number(int min, int max){                                    /* Returns a random FP number between min and max values. */
	return (float)floor((double)(min + (max - min + 1)*((float)rand()/(float)RAND_MAX)));
}

int 
perform_simple_check(const Matrix M){                                   /* Check for upper triangular matrix, that is, the principal diagonal elements are 1. */
	for(unsigned int i = 0; i < M.num_rows; i++)
		if((fabs(M.elements[M.num_rows*i + i] - 1.0)) > 0.001) return 0;

	return 1;
} 


