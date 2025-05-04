#include "mathStats.h"

float math_geometric_mean(int n, float* numbers) {
    float product = 1;
    for (int i = 0; i < n; i++) {
      product *= numbers[i];
    }
    return pow(product, 1.0/n);
  }
  
  float math_variance(int n, float* numbers, float mean) {
    float sum_squared_differences = 0;
    for (int i = 0; i < n; i++) {
      sum_squared_differences += pow(numbers[i] - mean, 2);
    }
    return sum_squared_differences / (n - 1);
  }