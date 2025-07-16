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

#define SHARED_BUFFER_SIZE      4096
#define SHARED_MEMORY_DIM       (SHARED_BUFFER_SIZE*sizeof(float))
#define WARP_MAX                128
#define ROUND_SIZE              256

__inline__ __device__ float reduceSum(float value) {
    value += __shfl_down_sync(0xFFFFFFFF, value , 16);
    value += __shfl_down_sync(0xFFFFFFFF, value , 8);
    value += __shfl_down_sync(0xFFFFFFFF, value , 4);
    value += __shfl_down_sync(0xFFFFFFFF, value , 2);
    value += __shfl_down_sync(0xFFFFFFFF, value , 1);
    return value;
}

__inline__ __device__ int binarySearch(unsigned int* array, unsigned int lastElement, unsigned int value,unsigned int laneId) {
    int row=-1;
    unsigned int left=0;
    unsigned int right=lastElement;
    if (array[left]<=value && array[right]>=value) {    
        while (left<=right) {
            int mid = (left+right) / 2;
            if (array[mid] <= value) {
                row = mid;                    
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }
    return row;
}

__global__ void vmcsr_mul(Vector* output,SparseMatrix* matrix,Vector* input,unsigned int* blockDivision) {        
    float* __restrict__ cachedInput=input->dataArray; 
    extern __shared__ unsigned int loadedRows[];
    float* rowsBuffer=(float*) (loadedRows+blockDim.x);
    float* sharedBuffer=rowsBuffer+blockDim.x;
    unsigned int laneId=threadIdx.x%warpSize;

    for (unsigned int baseRow=blockDivision[blockIdx.x];baseRow<blockDivision[blockIdx.x+1];baseRow+=blockDim.x-1) {
        unsigned int rowLimit=min(blockDivision[blockIdx.x+1]-baseRow,(blockDim.x-1));
        if (threadIdx.x<=rowLimit) {
            rowsBuffer[threadIdx.x]=0;
            loadedRows[threadIdx.x]=matrix->rowArray[baseRow+threadIdx.x];
        }
        __syncthreads();

        for (unsigned int bufferPos=loadedRows[0];bufferPos<loadedRows[rowLimit];bufferPos+=SHARED_BUFFER_SIZE) {
            unsigned int bufferLimit=min(SHARED_BUFFER_SIZE ,loadedRows[rowLimit]-bufferPos);
            for (unsigned int i=threadIdx.x;i<bufferLimit;i+=blockDim.x) {
                sharedBuffer[i]=matrix->dataArray[bufferPos+i]*__ldg(&cachedInput[matrix->colArray[bufferPos+i]]);
            };
            __syncthreads();
            unsigned int elementsPerWarp=(bufferLimit+(blockDim.x/warpSize)-1)/(blockDim.x/warpSize);
            unsigned int warpStart=elementsPerWarp*(threadIdx.x/warpSize);
            unsigned int warpEnd=min(warpStart+elementsPerWarp,bufferLimit);
            unsigned int row=binarySearch(loadedRows,rowLimit,(warpStart+bufferPos),laneId);
            unsigned int endRow=min(warpEnd,loadedRows[row+1]-bufferPos);
            float acc=0;
            while(warpStart<warpEnd) {
                while(warpStart>=endRow) {
                    acc=reduceSum(acc);
                    if (laneId==0 && acc!=0) atomicAdd(&rowsBuffer[row],acc); 
                    acc=0;
                    row++;
                    endRow=min(warpEnd,loadedRows[row+1]-bufferPos);
                    
                }
                unsigned int index=min(warpStart+laneId,endRow-1);
                acc+=sharedBuffer[index]*(warpStart+laneId<endRow && warpStart<warpEnd);
                warpStart+=min(warpSize,endRow-warpStart);
                
            }
            acc=reduceSum(acc);
            if (laneId==0 && acc!=0) atomicAdd(&rowsBuffer[row],acc);
        }
        __syncthreads();
        if (threadIdx.x<rowLimit) {
            output->dataArray[baseRow+threadIdx.x]=rowsBuffer[threadIdx.x];
        }
    }
}

__global__ void vmcsr_div(SparseMatrix* matrix,unsigned int* blockDivision) {    
    unsigned int elementsPerBlock = (matrix->notNull + gridDim.x - 1) / gridDim.x;
    unsigned int blockStart = blockIdx.x * elementsPerBlock;
    unsigned int blockEnd = min(blockStart + elementsPerBlock, matrix->notNull);

    if(blockIdx.x==0) {
        blockDivision[0]=0;
        blockDivision[gridDim.x]=matrix->rowSize;
        return;
    }

    int row=-1;
    unsigned int left=0;
    unsigned int right=matrix->rowSize;
    if (matrix->rowArray[left]<=blockStart && matrix->rowArray[right]>=blockStart) {        
        while (left<=right) {
            int mid = (left+right) / 2;
            if (matrix->rowArray[mid] <= blockStart) {
                row = mid;                    
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }
    blockDivision[blockIdx.x]=row;
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

    if (threads<=0 || blocks<=0) {
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
    float* buffer=NULL;
    cudaResult=cudaMallocManaged((void**)&buffer, matrix->notNull*sizeof(float));
        if (cudaResult != cudaSuccess) {
        fprintf(stderr, "Error during buffer vector creation: %s\n",cudaGetErrorString(cudaResult));
        return cudaResult;
    }
    unsigned int* blockDivision;
    cudaResult=cudaMallocManaged((void**)&blockDivision, (blocks+1)*sizeof(unsigned int));
        if (cudaResult != cudaSuccess) {
        fprintf(stderr, "Error during buffer vector creation: %s\n",cudaGetErrorString(cudaResult));
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
        vmcsr_div<<<blocks,1>>>(matrix,blockDivision);
        
        if (i>=0) cudaEventRecord(stop);
        cudaResult=cudaEventSynchronize(stop); 
        
        if (cudaResult != cudaSuccess) {
            fprintf(stderr, "Error during kernel execution: %s\n", cudaGetErrorString(cudaResult));
            return cudaResult;
        }
        if (i>=0) {
            cudaEventElapsedTime(&times[i], start,stop);
        }
        if (i>=0) cudaEventRecord(start);
        unsigned int sharedDim=SHARED_MEMORY_DIM+(sizeof(float)*threads)+(sizeof(unsigned int)*threads);
        vmcsr_mul<<<blocks,threads,sharedDim>>>(output,matrix,vector,blockDivision);
        
        if (i>=0) cudaEventRecord(stop);
        cudaResult=cudaEventSynchronize(stop); 
        
        if (cudaResult != cudaSuccess) {
            fprintf(stderr, "Error during kernel execution: %s\n", cudaGetErrorString(cudaResult));
            return cudaResult;
        }
        if (i>=0) {
            float time=times[i];
            cudaEventElapsedTime(&times[i], start,stop);
            times[i]+=time;
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
    bool one=false;
    for (int i=0;i<output->size;i++) {
        if (output->dataArray[i] != outputSequential->dataArray[i]) {
            if (!one) {
                printf("Mismatch on %d\n",i);
                one=true;
            }
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

