#include <stdio.h>
#include <stdlib.h>


int main(int argc, char** argv) {
    int devices = 0;
    cudaError_t error_id = cudaGetDeviceCount(&devices);
    if (error_id != cudaSuccess) {
        printf("Error: No CUDA capable device found\n");
        return 1;
    }
    printf("Found %d devices with CUDA capabilities!\n", devices);
    int dev, driverVersion = 0, runtimeVersion = 0;

    for (dev = 0; dev < devices; ++dev) {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
            driverVersion / 1000, (driverVersion % 100) / 10,
            runtimeVersion / 1000, (runtimeVersion % 100) / 10);
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
            deviceProp.major, deviceProp.minor);

        printf("  Available global memory:                       %f Mbytes \n",(deviceProp.totalGlobalMem / 1048576.0f));
        
        printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)\n",deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
        printf("  Memory Clock rate:                             %.0f Mhz\n",deviceProp.memoryClockRate * 1e-3f);
        printf("  Memory Bus Width:                              %d-bit\n",deviceProp.memoryBusWidth);
     
        if (deviceProp.l2CacheSize) {
           printf("  L2 Cache Size:                                 %d bytes\n",deviceProp.l2CacheSize);
        }   
        
        printf("  Total amount of shared memory per block:       %zu bytes\n",
               deviceProp.sharedMemPerBlock);
        printf("  Total shared memory per multiprocessor:        %zu bytes\n",
               deviceProp.sharedMemPerMultiprocessor);
        
        printf("  Maximum number of threads per multiprocessor:  %d\n",
               deviceProp.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n",
               deviceProp.maxThreadsPerBlock);
        printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
       
    
    }
    return 0;
}