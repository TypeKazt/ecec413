/* Matrix multiplication: C = A * B.
 * Host code.
 * 
 * Author: Naga Kandasamy
 * Date: 02/14/2017
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

// includes, kernels
#include "matrixmul_kernel.cu"

extern "C" void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);
Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width, int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);
void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P);
void checkCUDAError(const char *msg);
int checkResults(float *, float *, int, float);

int 
main(int argc, char** argv) 
{
    Matrix  M, N, P;
	
    srand(52);  /* Change to srand(time(NULL)) if you want a different matrix each time. */

    M  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 1);	        /* Allocate and initialize the matrices. */
	N  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 1);
	P  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 0);

	printf("Multiplying matrices on the GPU \n");
    MatrixMulOnDevice(M, N, P);                         	    /* Compute M * N on the device. */

    printf("Multiplying matrices on the CPU. \n");
    struct timeval start, stop;	
	gettimeofday(&start, NULL);

    Matrix reference = AllocateMatrix(P.height, P.width, 0);    /* Compute M * N on the CPU. */
	computeGold(reference.elements, M.elements, N.elements, M.height, M.width, N.width);

    gettimeofday(&stop, NULL);
	printf("Execution time = %fs. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));

		
	/* Check if the device result is equivalent to the expected solution. */
    int num_elements = M.height*M.width;
	int status = checkResults(reference.elements, P.elements, num_elements, 0.001f);
	printf("Test %s\n", (1 == status) ? "PASSED" : "FAILED");
	
	/* Free matrices. */
	FreeMatrix(&M);
	FreeMatrix(&N);
	FreeMatrix(&P);
	return 0;
}


void 
MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P)
{
    Matrix Md = AllocateDeviceMatrix(M);                    /* Load M and N to the device. */
	CopyToDeviceMatrix(Md, M);
	Matrix Nd = AllocateDeviceMatrix(N);
	CopyToDeviceMatrix(Nd, N);

    Matrix Pd = AllocateDeviceMatrix(P);                    /* Allocate P on the device. */


    dim3 threads(TILE_SIZE, TILE_SIZE);                     /* Set up the execution grid. */

    printf("Setting up a %d x %d grid of thread blocks. \n", (Pd.width + TILE_SIZE - 1)/TILE_SIZE,\\
            (Pd.height + TILE_SIZE - 1)/TILE_SIZE);

	dim3 grid((Pd.width + TILE_SIZE - 1)/TILE_SIZE, (Pd.height + TILE_SIZE - 1)/TILE_SIZE);

    struct timeval start, stop;	
	gettimeofday(&start, NULL);

	/* Lxecute the kernel. */
	MatrixMulKernel<<< grid, threads >>>(Pd.elements, Md.elements, Nd.elements, MATRIX_SIZE);
	cudaThreadSynchronize();

    gettimeofday(&stop, NULL);
	printf("Execution time = %fs. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));

    checkCUDAError("Error in kernel");                  /* Check if kernel execution generated an error. */

    CopyFromDeviceMatrix(P, Pd);                        /* Read P from the device. */

	
    FreeDeviceMatrix(&Md);                                  /* Free device matrices. */
	FreeDeviceMatrix(&Nd);
	FreeDeviceMatrix(&Pd);
}

Matrix 
AllocateDeviceMatrix(const Matrix M)                        /* Allocate a device matrix of same size as M. */
{
	Matrix Mdevice = M;
	int size = M.width * M.height * sizeof(float);
	cudaMalloc((void**)&Mdevice.elements, size);
	return Mdevice;
}

/* Allocate a device matrix of dimensions height*width
   If init == 0, initialize to all zeroes.  
   If init == 1, perform random initialization.
   */
Matrix 
AllocateMatrix(int height, int width, int init)
{
	Matrix M;
	M.width = M.pitch = width;
	M.height = height;
	int size = M.width * M.height;
	M.elements = NULL;
	
	M.elements = (float*) malloc(size*sizeof(float));

	for(unsigned int i = 0; i < M.height * M.width; i++){
		M.elements[i] = (init == 0) ? (0.0f) : floor((rand()*3 / (float)RAND_MAX));
	}
	return M;
}	

void 
CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)      /* Copy a host matrix to a device matrix. */
{
	int size = Mhost.width * Mhost.height * sizeof(float);
	Mdevice.height = Mhost.height;
	Mdevice.width = Mhost.width;
	Mdevice.pitch = Mhost.pitch;
	cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
}

void 
CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)    /* Copy a device matrix to a host matrix. */
{
	int size = Mdevice.width * Mdevice.height * sizeof(float);
	cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
}

void 
FreeDeviceMatrix(Matrix* M)                                 /* Free a device matrix. */
{
	cudaFree(M->elements);
	M->elements = NULL;
}

// Free a host Matrix
void 
FreeMatrix(Matrix* M)
{
	free(M->elements);
	M->elements = NULL;
}


void 
checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		printf("CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}						 
}


int 
checkResults(float *reference, float *gpu_result, int num_elements, float threshold)
{
    int checkMark = 1;
    float epsilon = 0.0;
    
    for(int i = 0; i < num_elements; i++)
        if(fabsf((reference[i] - gpu_result[i])/reference[i]) > threshold){
            checkMark = 0;
        }

    for(int i = 0; i < num_elements; i++)
        if(fabsf((reference[i] - gpu_result[i])/reference[i]) > epsilon){
            epsilon = fabsf((reference[i] - gpu_result[i])/reference[i]);
        }

    printf("Max epsilon = %f. \n", epsilon); 
    return checkMark;
}
