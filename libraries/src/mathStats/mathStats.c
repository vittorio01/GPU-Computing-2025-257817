#include "mathStats.h"

double math_geometric_mean(int n, float* numbers) {
    double product = 1;
    for (int i = 0; i < n; i++) {
      product *= (double) numbers[i];
    }
    return pow(product, 1.0/n);
  }
  
double math_variance(int n, float* numbers, float mean) {
    double sum_squared_differences = 0;
    for (int i = 0; i < n; i++) {
      sum_squared_differences += pow(((double)numbers[i]) - mean, 2);
    }
    return sum_squared_differences / (n - 1);
  }