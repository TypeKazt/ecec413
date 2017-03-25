#include <stdio.h>
#include <math.h>
#include <float.h>

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C" void compute_gold(int *, int *, int, int);

void compute_gold(int *input_data, int *histogram, int num_elements, int histogram_size){
  int i;
  
  // Initialize histogram
  for(i = 0; i < histogram_size; i++) 
			 histogram[i] = 0; 

  // Bin the elements in the input stream
  for(i = 0; i < num_elements; i++)
			 histogram[input_data[i]]++;
}

