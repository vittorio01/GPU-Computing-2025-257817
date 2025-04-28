#ifndef __MATH_STATS_H__ 
#define __MATH_STATS_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>

double math_geometric_mean(int n, double* numbers);
double math_variance(int n, double* numbers,double mean);

#ifdef __cplusplus
}
#endif

#endif