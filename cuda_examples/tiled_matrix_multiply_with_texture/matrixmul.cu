/* Matrix multiplication: C = A * B. Host side code. 

Author: Naga Kandasamy
Date modified: 03/07/2017

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
Matrix AllocateDeviceMatrix(const Matrix);
Matrix AllocateMatrix(int, int, int);
void CopyToDeviceMatrix(Matrix, const Matrix);
void CopyFromDeviceMatrix(Matrix, const Matrix);
void FreeDeviceMatrix(Matrix *);
void FreeMatrix(Matrix *);
void MatrixMulOnDevice(const Matrix, const Matrix, Matrix);
void checkCUDAError(const char *);
int checkResults(float *, float *, int, float);


int 
main(int argc, char** argv) {

	Matrix  M;
	Matrix  N;
	Matrix  P;
	
	srand(time(NULL));
	
	// Allocate and initialize the matrices
	M  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 1);	
	N  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 1);
	P  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 0);

	// M * N on the device
    printf("Performing matrix multiplication on the GPU. \n");
	MatrixMulOnDevice(M, N, P);	
	printf("GPU computation complete. \n");

	// compute the matrix multiplication on the CPU for comparison
	Matrix reference = AllocateMatrix(P.height, P.width, 0);
    printf("Performing matrix multiplication on the CPU. \n");
	computeGold(reference.elements, M.elements, N.elements, M.height, M.width, N.width);
	printf("CPU computation complete. \n");
	
    /* Check if the device result is equivalent to the expected solution. */
    int num_elements = P.height*P.width;
	int status = checkResults(reference.elements, P.elements, num_elements, 0.001f);
	printf("Test %s\n", (1 == status) ? "PASSED" : "FAILED");

	// Free matrices
	FreeMatrix(&M);
	FreeMatrix(&N);
	FreeMatrix(&P);
	return 0;
}


void 
MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P)
{
	// Load M and N to the device
    Matrix Md = AllocateDeviceMatrix(M);
	CopyToDeviceMatrix(Md, M);
	Matrix Nd = AllocateDeviceMatrix(N);
	CopyToDeviceMatrix(Nd, N);

	// Allocate P on the device
	Matrix Pd = AllocateDeviceMatrix(P);
	
	/* Bind Md and Nd to 1D textures. Note: the maximum width for 1D texture reference bound to linear memory
       varies with the GPU generation and compute capability. Currently it is set to 2^{27} elements. */ 
	// cudaBindTexture(NULL, M_on_tex, Md.elements, M.width * M.height * sizeof(float));
	// cudaBindTexture(NULL, N_on_tex, Nd.elements, N.width * N.height * sizeof(float));

	/* Bind Md and Nd to 2D textures. Note: as with 1D textures, there is a maximum width and height 
    for 2D texture reference bound to a CUDA array or to a linear memory. */
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	cudaBindTexture2D(NULL, M_on_tex_2D, Md.elements, desc, M.width, M.height, M.width * sizeof(float));
	cudaBindTexture2D(NULL, N_on_tex_2D, Nd.elements, desc, N.width, N.height, N.width * sizeof(float));

	// Setup the execution configuration
	dim3 threads(TILE_SIZE, TILE_SIZE);
	dim3 grid((Pd.width + TILE_SIZE - 1)/TILE_SIZE, (Pd.height + TILE_SIZE - 1)/TILE_SIZE);

    struct timeval start, stop;	
	gettimeofday(&start, NULL);

	// Execute the kernel
	// MatrixMulKernel_vanilla<<< grid, threads >>>(Pd.elements, Md.elements, Nd.elements, MATRIX_SIZE);
	// MatrixMulKernel_1Dtex<<< grid, threads >>>(Pd.elements, Md.elements, Nd.elements, MATRIX_SIZE);
	MatrixMulKernel_2Dtex<<< grid, threads >>>(Pd.elements, Md.elements, Nd.elements, MATRIX_SIZE);

	cudaThreadSynchronize();

    // check if kernel execution generated an error
	checkCUDAError("Error in kernel");

    gettimeofday(&stop, NULL);
	printf("Execution time = %fs. \n",\
            (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));

	// Read P from the device
	CopyFromDeviceMatrix(P, Pd); 

	// Unbind texture references
	// cudaUnbindTexture(M_on_tex);
	// cudaUnbindTexture(N_on_tex);

	cudaUnbindTexture(M_on_tex_2D);
	cudaUnbindTexture(N_on_tex_2D);

	// Free device matrices
	FreeDeviceMatrix(&Md);
	FreeDeviceMatrix(&Nd);
	FreeDeviceMatrix(&Pd);
}

// Allocate a device matrix of same size as M.
Matrix 
AllocateDeviceMatrix(const Matrix M)
{
	Matrix Mdevice = M;
	int size = M.width * M.height * sizeof(float);
	cudaMalloc((void**)&Mdevice.elements, size);
	return Mdevice;
}

// Allocate a device matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
//  If init == 2, initialize matrix parameters, but do not allocate memory 
Matrix 
AllocateMatrix(int height, int width, int init)
{
	Matrix M;
	M.width = M.pitch = width;
	M.height = height;
	int size = M.width * M.height;
	M.elements = NULL;
	
	// don't allocate memory on option 2
	if(init == 2)
		return M;
		
	M.elements = (float*) malloc(size*sizeof(float));

	for(unsigned int i = 0; i < M.height * M.width; i++){
		M.elements[i] = (init == 0) ? (0.0f) : (rand()*3 / (float)RAND_MAX);
	}
	return M;
}	

// Copy a host matrix to a device matrix.
void 
CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
	int size = Mhost.width * Mhost.height * sizeof(float);
	Mdevice.height = Mhost.height;
	Mdevice.width = Mhost.width;
	Mdevice.pitch = Mhost.pitch;
	cudaMemcpy(Mdevice.elements, Mhost.elements, size, 
					cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void 
CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
	int size = Mdevice.width * Mdevice.height * sizeof(float);
	cudaMemcpy(Mhost.elements, Mdevice.elements, size, 
					cudaMemcpyDeviceToHost);
}

// Free a device matrix.
void 
FreeDeviceMatrix(Matrix* M)
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
            break;
        }

    for(int i = 0; i < num_elements; i++)
        if(fabsf((reference[i] - gpu_result[i])/reference[i]) > epsilon){
            epsilon = fabsf((reference[i] - gpu_result[i])/reference[i]);
        }

    printf("Max epsilon = %f. \n", epsilon); 
    return checkMark;
}
