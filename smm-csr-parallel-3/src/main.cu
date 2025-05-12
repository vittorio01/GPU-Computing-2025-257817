#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "math.h"
#include "dataLib.h"
#include "mathStats.h"

#define MISSING_MATRIX_INPUT_ERROR 10
#define DATA_TRANSFER_ERROR 11

#define WARMUP_CYCLES 5
#define ITERATIONS 20

#define DEFAULT_THREADS_NUMBER  1
#define DEFAULT_BLOCKS_NUMBER   1

__global__ void vmcsr_mul(Vector* output, SparseMatrix* matrix,Vector* input) {    
    int rowsPerBlock = (matrix->rowSize + gridDim.x - 1) / gridDim.x;
    int blockStart = blockIdx.x * rowsPerBlock;
    int blockEnd = min(blockStart + rowsPerBlock, matrix->rowSize);
    if (rowsPerBlock == 0) return;

    for (int i=blockStart+threadIdx.x;i<blockEnd;i+=blockDim.x) {
        output->dataArray[i]=0;
    }
    __syncthreads();

    for (int selectedRowIndex=blockStart;selectedRowIndex<blockEnd;selectedRowIndex++) {

        int startRow=matrix->rowArray[selectedRowIndex];
        int endRow=matrix->rowArray[selectedRowIndex+1];
        int elements=endRow-startRow;
        if (blockDim.x<elements) {
            float acc=0;
            int colPerThread = (elements + blockDim.x - 1) / blockDim.x;
            int threadStart = threadIdx.x * colPerThread;
            int threadEnd = min(threadStart + colPerThread, elements);
            threadStart+=startRow;
            threadEnd+=startRow;
            for (int j=threadStart;j<threadEnd;j++) {            
                acc+=matrix->dataArray[j]*input->dataArray[matrix->colArray[j]];
            }
            atomicAdd(&output->dataArray[selectedRowIndex],acc);
        } else {
            if (threadIdx.x<elements) {
                atomicAdd(&output->dataArray[selectedRowIndex],matrix->dataArray[startRow+threadIdx.x]*input->dataArray[matrix->colArray[startRow+threadIdx.x]]);
            }
        }
    }
}


void vmcsr_mul_sequential(Vector* output, SparseMatrix* matrix,Vector* vector) {
    output->size=matrix->rowSize;  
    for (int i=0;i<(matrix->rowSize);i++) {
        float acc=0;
        for (int j=(matrix->rowArray[i]);j<(matrix->rowArray[i+1]);j++) {
            acc+=matrix->dataArray[j]*vector->dataArray[matrix->colArray[j]];
        }
        output->dataArray[i]=acc;
    }
}


int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Missing input matrix. Closing program...\n");
        return MISSING_MATRIX_INPUT_ERROR;
    }
    int blocks=DEFAULT_BLOCKS_NUMBER;
    int threads=DEFAULT_THREADS_NUMBER;

    for (int i=2;i<argc;i++) {
        if (strcmp(argv[i],"-b")==0 && (i+1)<argc) {
            i++;
            blocks=atoi(argv[i]);
            continue;
        }
        if (strcmp(argv[i],"-t")==0 && (i+1)<argc) {
            i++;
            threads=atoi(argv[i]);
            continue;
        }
    }

    if (threads<0 || blocks<0) {
        printf("Invalid format of the blocks/threads organization: %d blocks, %d threads\n",blocks,threads);
        return 1;
    }
    printf("Launching algorithm with %d blocks and %d threads\n",blocks,threads);
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
        vmcsr_mul<<<blocks,threads>>>(&output,&matrix,&vector);
        
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

    double bandwidth=(((12*matrix.notNull)+(16*matrix.rowSize))/mean_value)*pow(10,-6);
    printf("Effective bandwidth: %2f GB/s\n",bandwidth);

    printf("checking results...\n");
    Vector outputSequential;
    vectorCreate(&outputSequential,matrix.rowSize);
    vmcsr_mul_sequential(&outputSequential,&matrix,&vector);
    int mismatches=0;
    float maxEpsilon=0;
    for (int i=0;i<output.size;i++) {
        if (output.dataArray[i] != outputSequential.dataArray[i]) {
            mismatches++;
            float epsilon=fabs(output.dataArray[i] - outputSequential.dataArray[i]);
            if (maxEpsilon<epsilon) maxEpsilon=epsilon;
        } else {
            
        }
        
    }
    if (mismatches>0) {
        printf("Found %d mismatches with max epsilon %f . Please check if your algorithm works\n",mismatches,maxEpsilon);
    } else {
        printf("The algorithm works good :)\n");
    }

    printf("Operation done. Cleaning heap and VRAM...\n");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    vectorDestroy(&vector);
    vectorDestroy(&output);
    vectorDestroy(&outputSequential);
    matrixDestroy(&matrix);
    
    return 0;
}   

