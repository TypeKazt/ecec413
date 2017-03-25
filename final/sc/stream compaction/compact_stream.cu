#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

// includes, kernels
#include "compact_stream_kernel.cu"

#define NUM_ELEMENTS 1024

void compact_stream(void);
extern "C" unsigned int compare( const float* reference, const float* data, const unsigned int len);
extern "C" void compute_scan_gold( float* reference, float* idata, const unsigned int len);
extern "C" int compact_stream_gold(float *reference, float *idata, unsigned int len);
int compact_stream_on_device(float *result_d, float *h_data, unsigned int num_elements);
int checkResults(float *reference, float *result_d, int num_elements, float threshold);


int main( int argc, char** argv) 
{
    compact_stream();
    exit(0);
}

void compact_stream(void) 
{
    unsigned int num_elements = NUM_ELEMENTS;
    const unsigned int mem_size = sizeof(float) * num_elements;

    // allocate host memory to store the input data
    float* h_data = (float *) malloc(mem_size);
      
    // initialize the input data on the host to be integer values
    // between 0 and 1000, both positive and negative
	 srand(time(NULL));
	 float rand_number;
     for( unsigned int i = 0; i < num_elements; ++i) {
         rand_number = rand()/(float)RAND_MAX;
         if(rand_number > 0.5) 
             h_data[i] = floorf(1000*(rand()/(float)RAND_MAX));
         else 
             h_data[i] = -floorf(1000*(rand()/(float)RAND_MAX));
     }


    /* Compute reference solution. The function compacts the stream and stores the 
       length of the new steam in num_elements. */
    float *reference = (float *) malloc(mem_size);  
    int stream_length_cpu;
    stream_length_cpu = compact_stream_gold(reference, h_data, num_elements);

  	/* Add your code to perform the stream compaction on the GPU. 
       Store the result in gpu_result. */
    float *result_d = (float *) malloc(mem_size);
    int stream_length_d;
    stream_length_d = compact_stream_on_device(result_d, h_data, num_elements);

	// Compare the reference solution with the GPU-based solution
    int res = checkResults(reference, result_d, stream_length_cpu, 0.0f);
    printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");

    // cleanup memory
    free(h_data);
    free(reference);
}

// Use the GPU to compact the h_data stream 
int compact_stream_on_device(float *result_d, float *h_data, unsigned int num_elements)
{
    int n = 0; // Number of elements in the compacted stream

    return n;
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
