#include <iostream>
#include <math.h>
__global__
void add(int n, float *x, float *y) {
    for (int i = 0; i < n; i++) y[i] = x[i] + y[i];
}
int main(void) {
    int N = 256;
    float *x, *y;

    cudaError_t err = cudaMallocManaged(&x, N*sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Errore in cudaMallocManaged: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    cudaMallocManaged(&y, N*sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    add<<<1, 1>>>(N, x, y);

    cudaDeviceSynchronize();

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    cudaFree(x);
    cudaFree(y);
}