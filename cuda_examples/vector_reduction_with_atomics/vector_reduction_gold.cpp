#include <stdio.h>
#include <math.h>
#include <float.h>

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C" float compute_gold( float *, unsigned int);


float compute_gold( float* A, unsigned int num_elements)
{
  unsigned int i;
  double sum = 0.0; 

  for(i = 0; i < num_elements; i++) 
	  sum += A[i];

  return (float)sum;
}







