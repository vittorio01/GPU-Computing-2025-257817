#include <stdio.h>
#include <stdlib.h>

__global__
void print_gpu() {
    printf("Hello\n");
}

int main() {

    print_gpu<<<1, 1>>>();

    cudaDeviceSynchronize();
    return 0;
}