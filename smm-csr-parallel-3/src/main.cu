#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>

#include "math.h"
#include "dataLib.h"
#include "mathStats.h"

#define MISSING_MATRIX_INPUT_ERROR 10
#define DATA_TRANSFER_ERROR 11

#define WARMUP_CYCLES 2
#define ITERATIONS 10

#define THREADS_NUMBER      1
#define BLOCK_NUMBER        1
#define SHARED_MEMORY_DIM   8 //49152 

__global__ void vmcsr_mul(Vector* output, SparseMatrix* matrix,Vector* input) {
    int threadNumber=(blockIdx.x*blockDim.x)+threadIdx.x;
    int threadShift=gridDim.x*blockDim.x;

    extern __shared__ double shared[];
    int sharedMemorySize=SHARED_MEMORY_DIM/sizeof(double);
    int sharedDataPerThread=sharedMemorySize/THREADS_NUMBER;
    for (int i=threadNumber;i<matrix->rowSize;i+=threadShift) {
        
        int startRow=matrix->rowArray[i];
        int endRow=matrix->rowArray[i+1];
        int elements=endRow-startRow;
        printf("thread %d computes for row %d with %d elements (%d %d)\n",threadNumber,i,elements,startRow,endRow);
        double acc=0;
        for (int windowPos=0;windowPos<input->size;windowPos+=sharedMemorySize) {
            printf("Window position -> %d\n",windowPos);
            //loading phase 
            for (int j=0;j<sharedDataPerThread;j++) {
                int sharedIndex=(threadIdx.x*sharedDataPerThread)+j;
                shared[sharedIndex]=input->dataArray[sharedIndex+windowPos];
            }
            __syncthreads();
            //computing phase
            if (elements>0) {
                for (int j=startRow+(endRow-elements);elements>0;j++) {
                    
                    int mxCol=matrix->colArray[j];
                    
                    if (mxCol>=windowPos+sharedMemorySize) break;
                    
                    printf("Computing elements on MXcol %d and shared position %d (%d left)\n",mxCol,mxCol-windowPos,elements);
                    acc+=matrix->dataArray[j]*shared[mxCol-windowPos];
                    elements--;
                
                }
            } 
            if (elements==0) {
                elements=-1;
                output->dataArray[i]=acc;
                printf("Result saved ->>> %d: row %d, value %2f\n",threadNumber,i,acc);
            
            }
            __syncthreads();
        
        }
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
        vmcsr_mul<<<THREADS_NUMBER,BLOCK_NUMBER,SHARED_MEMORY_DIM>>>(&output,&matrix,&vector);
        
        cudaEventRecord(stop);
        cudaResult=cudaEventSynchronize(stop); 
        if (cudaResult != cudaSuccess) {
            fprintf(stderr, "Error during kernel execution: %s\n", cudaGetErrorString(cudaResult));
            return cudaResult;
        }
        cudaEventElapsedTime(&times[i], start,stop);
        printf("iteration %d took %2f ms\n",i,times[i]);
        
        
        
    }
    

    float mean_value = math_geometric_mean(ITERATIONS,times);
    float variance = math_variance(ITERATIONS,times,mean_value);
    int floats=matrix.notNull;
    printf("Executed %d iterations, floating point operations: %d average time: %f ms variance: %f ms\n",ITERATIONS,matrix.notNull,(mean_value),(variance));
    double flops= (double)((double)floats)/(mean_value*pow(10,-3));
    printf("average GFLOP/s: %2f\n",flops*pow(10,-9));

    printf("Operation done. Cleaning heap and VRAM...\n");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    vectorDestroy(&vector);
    vectorDestroy(&output);
    matrixDestroy(&matrix);
    
    return 0;
}   

