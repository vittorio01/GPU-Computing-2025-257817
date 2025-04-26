#include <stdio.h>
#include <stdlib.h>

__global__ void printFromGPU(void) {
    printf("hello from GPU %d %d\n",threadIdx.x,blockIdx.x);
}

int main(void) {
    
    printFromGPU<<<10, 2>>>();

    cudaDeviceSynchronize();
    return 0;

}