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

#define DEFAULT_THREADS_NUMBER  1
#define DEFAULT_BLOCKS_NUMBER   1

/*
Implementation of the first algorithm for the parallel SpMV multiplication for CSR matrices. 

The row vector divided for the number of block used for the multiplication (if the division is not perfect the last block obtains less rows to scan).
The same operation is made to assign a certain number of rows in each subspace to all threads in all blocks.

Each thread performs the multiplication and accumulation for a certain number of rows and saves the result in the output row. 

*/
__global__ void vmcsr_mul(Vector* output, SparseMatrix* matrix,Vector* input) {
    //row vector division in block subspaces
    int rowsPerBlock = (matrix->rowSize + gridDim.x - 1) / gridDim.x;
    int blockStart = blockIdx.x * rowsPerBlock;
    int blockEnd = min(blockStart + rowsPerBlock, matrix->rowSize);
    
    if (rowsPerBlock == 0) return; 

    //row vector supspace division for all threads
    int rowsPerThread = (rowsPerBlock + blockDim.x - 1) / blockDim.x;
    int threadStart = threadIdx.x * rowsPerThread;
    int threadEnd = min(threadStart + rowsPerThread, rowsPerBlock);
   
    threadStart+=blockStart;
    threadEnd+=blockStart;

    for (int selectedRowIndex=threadStart;selectedRowIndex<threadEnd;selectedRowIndex++) {
        int startRow=matrix->rowArray[selectedRowIndex];
        int endRow=matrix->rowArray[selectedRowIndex+1];
        float acc = 0;

        //each row is assigned at one thread 
        for (int i=startRow;i<endRow;i++) {
            acc+=matrix->dataArray[i]*input->dataArray[matrix->colArray[i]];
        }
        output->dataArray[selectedRowIndex]=acc;
    }
}

//SpMV multiplication algorithm used for checking the results. 
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
    //Data loading phase: the programs checks if there is a matrix, opens it using the datalib custom library and creates a vector for the multiplication
    if (argc < 2) {
        printf("Missing input matrix. Closing program...\n");
        return MISSING_MATRIX_INPUT_ERROR;
    }

    //verify the arguments for the blocks and threads allocation  
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
        }
    }

    if (threads<0 || blocks<0) {
        printf("Invalid format of the blocks/threads organization: %d blocks, %d threads\n",blocks,threads);
        return 1;
    }
    printf("Launching algorithm with %d blocks and %d threads\n",blocks,threads);

    //Creating matrix structure and opening the file
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
    
    //Creating input and output vector
    printf("generating input and output vectors for the moltiplication... \n");
    Vector vector;
    vectorCreateRandom(&vector, matrix.colSize);
    Vector output;
    vectorCreate(&output,matrix.rowSize);

    //preloading all structures in the global memory 
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


     //Benchmarking phase: Using the function cudaEventRecord to measure the time used by the algorithm
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
    
    //Performing the results of the benchmarks for GFLOP/s and effective bandwidth
    double mean_value = math_geometric_mean(ITERATIONS,times);
    double variance = math_variance(ITERATIONS,times,mean_value);
    int floats=2*matrix.notNull;
    printf("Executed %d iterations, floating point operations: %d average time: %2f ms variance: %2f ms\n",ITERATIONS,matrix.notNull,(mean_value),(variance));
    double flops= ((double)floats)/(mean_value*pow(10,-3));
    printf("average GFLOP/s: %2f\n",flops*pow(10,-9));
    double bandwidth=(((12*matrix.notNull)+(12*matrix.rowSize))/mean_value)*pow(10,-6);
    printf("Effective bandwidth: %2f GB/s\n",bandwidth);
    

    //Data check phase: the parallel algorithm results are compared with the sequential ones.
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

    //Cleaning phase.
    printf("Operation done. Cleaning heap and VRAM...\n");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    vectorDestroy(&vector);
    vectorDestroy(&output);
    vectorDestroy(&outputSequential);
    matrixDestroy(&matrix);
    
    return 0;
}   

