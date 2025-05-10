#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>

#include "math.h"
#include "dataLib.h"
#include "mathStats.h"

#define MISSING_MATRIX_INPUT_ERROR 10
#define DATA_TRANSFER_ERROR 11

#define WARMUP_CYCLES 5
#define ITERATIONS 20

#define THREADS_NUMBER  1024
#define BLOCK_NUMBER    65535
#define SHARED_MEMORY_DIM   4

__global__ void vmcsr_mul(Vector* output, SparseMatrix* matrix,Vector* input) {
    extern __shared__ int sharedMemory[];
    int* nextPendingRow=sharedMemory;

    int rowToProcess=0;
    int blockStart;
    int blockEnd;
    if (matrix->rowSize<gridDim.x) {
        if (blockIdx.x<matrix->rowSize) {
            rowToProcess=1;
            blockStart=blockIdx.x;
        }
    } else {
        int blockSize=matrix->rowSize/gridDim.x;
        int remainder=matrix->rowSize%gridDim.x;
    
        if (blockIdx.x<remainder) {
            rowToProcess=blockSize+1;
            blockStart=blockIdx.x*rowToProcess;
        } else {
            rowToProcess=blockSize;
            blockStart=remainder*(blockSize+1)+(blockIdx.x-remainder)*blockSize;
        }
    }
    blockEnd=blockStart+rowToProcess;

    if (threadIdx.x==0) *nextPendingRow=0;
    __syncthreads();

    int selectedRowIndex;
    while(true) {
        selectedRowIndex=atomicAdd(nextPendingRow,1)+blockStart;
        if (selectedRowIndex>=blockEnd) break; 

        int startRow=matrix->rowArray[selectedRowIndex];
        int endRow=matrix->rowArray[selectedRowIndex+1];
        int acc = 0;
        for (int i=startRow;i<endRow;i++) {
            acc+=matrix->dataArray[i]*input->dataArray[matrix->colArray[i]];
        }
        output->dataArray[selectedRowIndex]=acc;
    }
}


int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Missing input matrix. Closing program...\n");
        return MISSING_MATRIX_INPUT_ERROR;
    }
    SparseMatrix matrix;
    int result=matrixOpen(argv[1],&matrix);
    switch (result) {
        case FILE_OPEN_ERROR:
        printf("Error during matrix reading phase: cannot open file\n");
        return FILE_OPEN_ERROR;

        case ROW_ARRAY_NOT_VALID:
        printf("Error during matrix reading phase: row array not valid\n");
        return ROW_ARRAY_NOT_VALID;

        case COL_ARRAY_NOT_VALID:
        printf("Error during matrix reading phase: column array not valid\n");
        return COL_ARRAY_NOT_VALID;

        case DATA_ARRAY_NOT_VALID:
        printf("Error during matrix reading phase: data array not valid\n");
        return DATA_ARRAY_NOT_VALID;

        case FILE_TRUNCATED:
        printf("Error during matrix reading phase: cannot open file\n");
        return FILE_TRUNCATED;

        default:
        printf("Matrix read successfully :)\n");
        break;
    }
    printf("Converting into CSR ... \n");
    matrixConvertCSR(&matrix);
    
    
    printf("generating input and output vectors for the moltiplication... \n");
    Vector vector;
    vectorCreateRandom(&vector, matrix.colSize);
    Vector output;
    vectorCreate(&output,matrix.rowSize);

    printf("copying data in the CUDA memory... \n");
    cudaError_t cudaResult;
    int currentDevice;
    cudaGetDevice(&currentDevice);
    cudaResult=matrixPrefetch(&matrix,currentDevice);
    if (cudaResult != cudaSuccess) {
        fprintf(stderr, "Error during matrix prefetching on device %d: %s\n", currentDevice,cudaGetErrorString(cudaResult));
        return cudaResult;
    }
    cudaResult=vectorPrefetch(&vector,currentDevice);
    if (cudaResult != cudaSuccess) {
        fprintf(stderr, "Error during input vector prefetching on device %d: %s\n", currentDevice,cudaGetErrorString(cudaResult));
        return cudaResult;
    }
    cudaResult=vectorPrefetch(&output,currentDevice);
    if (cudaResult != cudaSuccess) {
        fprintf(stderr, "Error during output vector prefetching on device %d: %s\n", currentDevice,cudaGetErrorString(cudaResult));
        return cudaResult;
    }
    printf("performing matrix to vector multiplication...\n");
    
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float times[ITERATIONS];
    
    for (int i=-WARMUP_CYCLES;i<ITERATIONS;i++) {
        if (i>=0) cudaEventRecord(start);
        vmcsr_mul<<<BLOCK_NUMBER,THREADS_NUMBER,SHARED_MEMORY_DIM>>>(&output,&matrix,&vector);
        
        if (i>=0) cudaEventRecord(stop);
        cudaResult=cudaEventSynchronize(stop); 
        if (cudaResult != cudaSuccess) {
            fprintf(stderr, "Error during kernel execution: %s\n", cudaGetErrorString(cudaResult));
            return cudaResult;
        }
        if (i>=0) {
            cudaEventElapsedTime(&times[i], start,stop);
            printf("iteration %d took %f ms\n",i,times[i]);
        }
    }
    

    double mean_value = math_geometric_mean(ITERATIONS,times);
    double variance = math_variance(ITERATIONS,times,mean_value);
    int floats=2*matrix.notNull;
    printf("Executed %d iterations, floating point operations: %d average time: %2f ms variance: %2f ms\n",ITERATIONS,matrix.notNull,(mean_value),(variance));
    double flops= ((double)floats)/(mean_value*pow(10,-3));
    printf("average GFLOP/s: %2f\n",flops*pow(10,-9));

    printf("Operation done. Cleaning heap and VRAM...\n");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    vectorDestroy(&vector);
    vectorDestroy(&output);
    matrixDestroy(&matrix);
    
    return 0;
}   

