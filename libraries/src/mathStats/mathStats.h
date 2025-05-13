//Lybrary that contains the definition of statistical functions used for benchmarking the algorithms

#ifndef __MATH_STATS_H__ 
#define __MATH_STATS_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdio.h>

/*math_geometric_mean performs the geometric mean of an array of floats:
- n -> size of the array
- numbers -> array of floats

returns -> geometric mean (double)
*/
double math_geometric_mean(int n, float* numbers);

/*math_variance performs the variance of an array of floats:
- n -> size of the array
- numbers -> array of floats
- mean -> the geometric mean of the floats

returns -> geometric mean (double)
*/
double math_variance(int n, float* numbers,float mean);

#ifdef __cplusplus
}
#endif

#endif