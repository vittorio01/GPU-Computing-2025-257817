#ifndef __MATH_STATS_H__ 
#define __MATH_STATS_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>

float math_geometric_mean(int n, float* numbers);
float math_variance(int n, float* numbers,float mean);

#ifdef __cplusplus
}
#endif

#endif