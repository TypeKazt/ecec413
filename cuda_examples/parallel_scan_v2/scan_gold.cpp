#include <stdio.h>
#include <math.h>
#include <float.h>

extern "C" void computeGold( float* reference, float* idata, const unsigned int len);

void
computeGold( float* reference, float* idata, const unsigned int len) 
{
  reference[0] = 0;
  double total_sum = 0;
  unsigned int i;
  for(i = 1; i < len; ++i){
      total_sum += idata[i-1];
      reference[i] = idata[i-1] + reference[i-1];
  }
  // Here it should be okay to use != because we have integer values
  // in a range where float can be exactly represented
  if (total_sum != reference[i-1])
      printf("Warning: exceeding single-precision accuracy.  Scan will be inaccurate.\n");
  
}


