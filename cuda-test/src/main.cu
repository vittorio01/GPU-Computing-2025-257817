#include <stdio.h>
#include <stdlib.h>

__global__
void print_gpu(void) {
    printf("Hello from %d %d\n",threadIdx.x,blockIdx.x);
}

int main() {
    printf("Starting GPU program");
    print_gpu<<<2, 1>>>();

    cudaDeviceSynchronize();
    return 0;
}