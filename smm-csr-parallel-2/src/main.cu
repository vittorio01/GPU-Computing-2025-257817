#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>

#include "math.h"
#include "dataLib.h"
#include "mathStats.h"

#define MISSING_MATRIX_INPUT_ERROR 10
#define DATA_TRANSFER_ERROR 11

#define WARMUP_CYCLES 2
#define ITERATIONS 1

#define THREADS_NUMBER      10 //256
#define BLOCK_NUMBER        10 //10
#define SHARED_MEMORY_DIM   49152       


__global__ void vmcsr_mul(Vector* output, SparseMatrix* matrix,Vector* input) {
    extern __shared__ float shared[];
    int tIdx = blockIdx.x * blockDim.x + threadIdx.x;

    int dataNumber=SHARED_MEMORY_DIM/(2*sizeof(float));
    int chunkSize=dataNumber/blockDim.x;

    float* sharedDataArray=shared;
    int* sharedColArray = (int*)(shared + (dataNumber));

    float* threadDataArray=sharedDataArray+(threadIdx.x*chunkSize);
    int* threadColArray=sharedColArray+(threadIdx.x*chunkSize);

    unsigned int rowNumber=(matrix->rowSize);
    unsigned int threadShift=gridDim.x*blockDim.x;

    for (int i=tIdx;i<rowNumber;i+=threadShift) {
        int startRow=matrix->rowArray[i];
        int endRow=matrix->rowArray[i+1];
        int elements=endRow-startRow;
        float acc=0;
        //processing in base of the dedicated shared memory chunk size
        for (int chunkPos=0;chunkPos<elements;chunkPos+=chunkSize) {
            int chunkElements=min(chunkSize, elements-chunkPos);
            //loading phase for a chunk of a row 
            for(int j=0;j<chunkElements;j++) {
                threadDataArray[j]=matrix->dataArray[startRow+chunkPos+j];
                threadColArray[j]=matrix->colArray[startRow+chunkPos+j];
            }
            //__syncthreads();
            //computing phase for a chunk of a row
            for(int j=0;j<chunkElements;j++) {
                acc+=threadDataArray[j]*input->dataArray[threadColArray[j]];
            }
            //__syncthreads();
        }
        output->dataArray[i]=acc; 
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
        cudaEventRecord(start);
        vmcsr_mul<<<BLOCK_NUMBER,THREADS_NUMBER,SHARED_MEMORY_DIM >>>(&output,&matrix,&vector);
        
        cudaEventRecord(stop);
        cudaResult=cudaEventSynchronize(stop); 
        if (cudaResult != cudaSuccess) {
            fprintf(stderr, "Error during kernel execution: %s\n", cudaGetErrorString(cudaResult));
            return cudaResult;
        }
        cudaEventElapsedTime(&times[i], start,stop);
        printf("iteration %d took %f ms\n",i,times[i]);
        
        
        
    }
    
    float mean_value = math_geometric_mean(ITERATIONS,times);
    float variance = math_variance(ITERATIONS,times,mean_value);
    int floats=matrix.notNull;
    printf("Executed %d iterations, floating point operations: %d average time: %f ms variance: %f ms\n",ITERATIONS,matrix.notNull,(mean_value),(variance));
    float flops= (float)((float)floats)/(mean_value*pow(10,-3));
    printf("average GFLOP/s: %f\n",flops*pow(10,-9));

    printf("Operation done. Cleaning heap and VRAM...\n");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    vectorDestroy(&vector);
    vectorDestroy(&output);
    matrixDestroy(&matrix);
    
    return 0;
}   

