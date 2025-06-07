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
#define SHARED_MEMORY_DIM   49152

#define ROW_BATCH_SIZE      8

/*
Implementation of the second algorithm for the parallel SpMV multiplication for CSR matrices. 

The row vector divided for the number of block used for the multiplication (if the division is not perfect the last block obtains less rows to scan).
This time for each block the respective threads contends the next pending row in a dynamic way:
1- An integer variable in the shared memory is initialized at 0 and used to identify the next pending row
2- each thread in a certain block uses the function atomicAdd to receive the value and automatically increment it for the next threads. 
3- If the readed value is valid (not greather that the rows to process per block) each thread performs the SpMV like the first algorithm, saves the result
   and jumps to the step 2.
*/
__inline__ __device__ float reduceSum(float value) {
    __syncwarp();
    value += __shfl_down_sync(0xFFFFFFFF, value , 16);
    value += __shfl_down_sync(0xFFFFFFFF, value , 8);
    value += __shfl_down_sync(0xFFFFFFFF, value , 4);
    value += __shfl_down_sync(0xFFFFFFFF, value , 2);
    value += __shfl_down_sync(0xFFFFFFFF, value , 1);
    return value;
}


__global__ void vmcsr_mul(Vector* output, SparseMatrix* __restrict__ matrix,Vector* __restrict__ input) {
    extern __shared__ unsigned int sharedMemory[];
    unsigned int* nextPendingRow=sharedMemory;

    float* sharedData=(float*) (nextPendingRow+1);

    unsigned short laneId = threadIdx.x % 32;  

    unsigned int sharedDataSize = (SHARED_MEMORY_DIM-sizeof(unsigned int))/(sizeof(float));
    unsigned int sharedBlockDim = (sharedDataSize + blockDim.x - 1)/blockDim.x;
    unsigned int sharedBlockStart = threadIdx.x * sharedBlockDim;
    unsigned int sharedBlockEnd = min(sharedBlockStart + sharedBlockDim, sharedDataSize); 

    //row vector division in block subspaces
    unsigned int rowsPerBlock = (matrix->rowSize + gridDim.x - 1) / gridDim.x;
    unsigned int blockStart = blockIdx.x * rowsPerBlock;
    unsigned int blockEnd = min(blockStart + rowsPerBlock, matrix->rowSize);
    rowsPerBlock=blockEnd-blockStart;
    
    if (rowsPerBlock == 0) return;

    unsigned int sharedPos=0;
    while(sharedPos<matrix->colSize) {
        __syncthreads();   
        //loading phase
        sharedBlockEnd=min(sharedBlockEnd,matrix->colSize - sharedPos);
        for (int i=sharedBlockStart;i<sharedBlockEnd;i++) {
            sharedData[i]=__ldg(input->dataArray[sharedPos+i]);
        }

        if (threadIdx.x==0) *nextPendingRow=0;

        __syncthreads();
        //execution phase
        unsigned int selectedRowIndex;
        if (laneId==0) {
            selectedRowIndex=atomicAdd(nextPendingRow,1)+blockStart;
        }
        selectedRowIndex=__shfl_sync(0xFFFFFFFF,selectedRowIndex, 0);
        while(selectedRowIndex<blockEnd) {
            int startRow=matrix->rowArray[selectedRowIndex];
            int endRow=matrix->rowArray[selectedRowIndex+1];
            int rowSize=endRow-startRow;
            float acc = 0;

            unsigned int elementsPerLane = (rowSize + warpSize -1) / warpSize;
            unsigned int laneStart = laneId * elementsPerLane + startRow;
            unsigned int laneEnd = min(laneStart + elementsPerLane, endRow);
            
            //add ballot sync with a mask
            for (unsigned int i=laneStart;i<laneEnd;i++) {
                unsigned int inputCol=matrix->colArray[i];
                int offset = inputCol - sharedPos;
                if (offset < sharedDataSize && offset>=0) acc += matrix->dataArray[i] * sharedData[offset];
            }
            acc=reduceSum(acc);
            
            if (laneId==0) {
                if (sharedPos==0) {
                    output->dataArray[selectedRowIndex]=acc;
                } else {
                    output->dataArray[selectedRowIndex]+=acc;
                }
                
                selectedRowIndex=atomicAdd(nextPendingRow,1)+blockStart;
            }
            selectedRowIndex=__shfl_sync(0xFFFFFFFF,selectedRowIndex, 0);
        }

        //step phase
        sharedPos+=sharedDataSize;
            
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
    printf("Launching algorithm with %d blocks and %d threads on matrix %s\n",blocks,threads,argv[1]);

    //Creating matrix structure and opening the file
    SparseMatrix* matrix=NULL;
    matrixCreate(&matrix);

    int result=matrixOpen(argv[1],matrix);
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

        case MEMORY_ALLOCATION_ERROR:
        printf("Error during matrix reading phase: error during memory allocation\n");
        return MEMORY_ALLOCATION_ERROR;

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
    cudaError_t cudaResult;

    printf("Converting into CSR ... \n");

    cudaResult=matrixConvertCSR(matrix);
    if (cudaResult != cudaSuccess) {
        fprintf(stderr, "Error during matrix conversion: %s\n",cudaGetErrorString(cudaResult));
        return cudaResult;
    }
    //Creating input and output vector
    printf("generating input and output vectors for the moltiplication... \n");
    Vector* vector=NULL;
    
    cudaResult=vectorCreateRandom(&vector, matrix->colSize);
    if (cudaResult != cudaSuccess) {
        fprintf(stderr, "Error during input vector creation: %s\n",cudaGetErrorString(cudaResult));
        return cudaResult;
    }
    Vector* output=NULL;
    cudaResult=vectorCreate(&output,matrix->rowSize);
    if (cudaResult != cudaSuccess) {
        fprintf(stderr, "Error during output vector creation: %s\n",cudaGetErrorString(cudaResult));
        return cudaResult;
    }

    //preloading all structures in the global memory 
    printf("copying data in the CUDA memory... \n");
    
    int currentDevice;
    cudaGetDevice(&currentDevice);
    cudaResult=matrixPrefetch(matrix,currentDevice);
    if (cudaResult != cudaSuccess) {
        fprintf(stderr, "Error during matrix prefetching on device %d: %s\n", currentDevice,cudaGetErrorString(cudaResult));
        return cudaResult;
    }
    cudaResult=vectorPrefetch(vector,currentDevice);
    if (cudaResult != cudaSuccess) {
        fprintf(stderr, "Error during input vector prefetching on device %d: %s\n", currentDevice,cudaGetErrorString(cudaResult));
        return cudaResult;
    }
    cudaResult=vectorPrefetch(output,currentDevice);
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
        vmcsr_mul<<<blocks,threads,SHARED_MEMORY_DIM>>>(output,matrix,vector);
        
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
    int floats=2*matrix->notNull;
    printf("Executed %d iterations, floating point operations: %d average time: %2f ms variance: %2f ms\n",ITERATIONS,floats,(mean_value),(variance));
    double flops= ((double)floats)/(mean_value*pow(10,-3));
    printf("average GFLOP/s: %2f\n",flops*pow(10,-9));
    double bandwidth=(((12*matrix->notNull)+(12*matrix->rowSize))/(mean_value))*pow(10,-6);
    printf("Effective bandwidth: %2f GB/s\n",bandwidth);
    

    //Data check phase: the parallel algorithm results are compared with the sequential ones.
    printf("checking results...\n");
    Vector* outputSequential=NULL;
    vectorCreate(&outputSequential,matrix->rowSize);
    vmcsr_mul_sequential(outputSequential,matrix,vector);
    int mismatches=0;
    float maxEpsilon=0;
    for (int i=0;i<output->size;i++) {
        if (output->dataArray[i] != outputSequential->dataArray[i]) {
            mismatches++;
            float epsilon=fabs(output->dataArray[i] - outputSequential->dataArray[i]);
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
    printf("Operation done. Cleaning heap and VRAM...\n\n");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    vectorDestroy(vector);
    vectorDestroy(output);
    vectorDestroy(outputSequential);
    matrixDestroy(matrix);
    
    return 0;
}   

