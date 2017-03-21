#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

// includes, kernels
#include "2Dconvolution_kernel.cu"

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int);

Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width, int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);
void ConvolutionOnDevice(const Matrix M, const Matrix N, Matrix P);
int checkResults(float *, float *, int, float);


int main(int argc, char** argv) 
{

	Matrix  A;
	Matrix  B;
	Matrix  C;
	
	srand(time(NULL));
	
	// Allocate and initialize the matrices
	A  = AllocateMatrix(KERNEL_SIZE, KERNEL_SIZE, 1);
	B  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 1);
	C  = AllocateMatrix(B.height, B.width, 0);

   struct timeval start, stop;
   gettimeofday(&start, NULL);

   /* Convolve matrix B with matrix A on the CPU. */
   Matrix reference = AllocateMatrix(C.height, C.width, 0);
   computeGold(reference.elements, A.elements, B.elements, B.height, B.width);
       
   gettimeofday(&stop, NULL);
   printf("CPU Execution time = %fs. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
 
	/* Convolve matrix B with matrix A on the device. */
   ConvolutionOnDevice(A, B, C);

   /* Check if the device result is equivalent to the expected solution. */
    int num_elements = C.height * C.width;
	int status = checkResults(reference.elements, C.elements, num_elements, 0.001f);
	printf("Test %s\n", (1 == status) ? "PASSED" : "FAILED");

   // Free matrices
   FreeMatrix(&A);
   FreeMatrix(&B);
   FreeMatrix(&C);
	
   return 0;
}


void ConvolutionOnDevice(const Matrix M, const Matrix N, Matrix P)
{

    // Load M and N to the device
    Matrix Md = AllocateDeviceMatrix(M);
    Matrix Nd = AllocateDeviceMatrix(N);
    // Allocate P on the device
    Matrix Pd = AllocateDeviceMatrix(P);

    // Setup the execution configuration
    dim3 grid((P.width + BLOCK_SIZE -1)/BLOCK_SIZE, (P.height + BLOCK_SIZE -1)/BLOCK_SIZE, 1);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    float gpu;
    cudaEvent_t gpu_start, gpu_end;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_end);
 
    cudaEventRecord(gpu_start, NULL);
    //CopyToDeviceMatrix(Md, M);
    cudaMemcpyToSymbol(sM, M.elements, M.width*M.height*sizeof(float));
    CopyToDeviceMatrix(Nd, N);
    CopyToDeviceMatrix(Pd, P); // Clear memory

    // Launch the device computation threads!
    ConvolutionKernel<<<grid, block>>>(Md, Nd, Pd);
    // Read P from the device
    CopyFromDeviceMatrix(P, Pd); 
    cudaEventRecord(gpu_end, NULL);
    cudaEventSynchronize(gpu_end);
    cudaEventElapsedTime(&gpu, gpu_start, gpu_end);

    // Free device matrices
    FreeDeviceMatrix(&Md);
    FreeDeviceMatrix(&Nd);
    FreeDeviceMatrix(&Pd);
    
    // // Load M and N to the device
    // Matrix Md = AllocateDeviceMatrix(M);
    // Matrix Nd = AllocateDeviceMatrix(N);

    // // Allocate P on the device
    // Matrix Pd = AllocateDeviceMatrix(P);

    // // Setup the execution configuration
    // dim3 grid((P.width + THREAD_BLOCK_SIZE -1)/THREAD_BLOCK_SIZE, (P.height + THREAD_BLOCK_SIZE -1)/THREAD_BLOCK_SIZE, 1);
    // dim3 block(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE, 1);

    // cudaMemcpyToSymbol(sM, M.elements, M.width*M.height*sizeof(float));
    // CopyToDeviceMatrix(Nd, N);
    // CopyToDeviceMatrix(Pd, P);

    // struct timeval start, stop;
    // gettimeofday(&start, NULL);

    // // Launch the device computation threads!
    // ConvolutionKernel<<<grid, block>>>(Md, Nd, Pd);
    // cudaThreadSynchronize();

    // gettimeofday(&stop, NULL);
    // printf("GPU Execution time = %fs. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));


    // // Read P from the device
    // CopyFromDeviceMatrix(P, Pd); 

    // // Free device matrices
    // FreeDeviceMatrix(&Md);
    // FreeDeviceMatrix(&Nd);
    // FreeDeviceMatrix(&Pd);

}

// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M)
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
Matrix AllocateMatrix(int height, int width, int init)
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
		M.elements[i] = (init == 0) ? (0.0f) : (rand() / (float)RAND_MAX);
		if(rand() % 2)
			M.elements[i] = - M.elements[i];
	}
    return M;
}	

// Copy a host matrix to a device matrix.
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.width * Mhost.height * sizeof(float);
    Mdevice.height = Mhost.height;
    Mdevice.width = Mhost.width;
    Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
    int size = Mdevice.width * Mdevice.height * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
}

// Free a device matrix.
void FreeDeviceMatrix(Matrix* M)
{
    cudaFree(M->elements);
    M->elements = NULL;
}

// Free a host Matrix
void FreeMatrix(Matrix* M)
{
    free(M->elements);
    M->elements = NULL;
}

// Check the CPU and GPU solutions
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
