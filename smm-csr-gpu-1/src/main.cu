#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>

#include "math.h"
#include "dataLib.h"
#include "mathStats.h"

#define MISSING_MATRIX_INPUT_ERROR 10
#define DATA_TRANSFER_ERROR 11

cudaError_t cudaVectorLoad(Vector* vector) {
    Vector cudaVector;
    cudaVector.size=vector->size;
    
    cudaError_t result;
    result=cudaMalloc((void**)&cudaVector.dataArray,(cudaVector.size*sizeof(double)));
    if (result!=cudaSuccess) return result;

    result=cudaMemcpy(cudaVector.dataArray,vector->dataArray,(cudaVector.size*sizeof(double)),cudaMemcpyHostToDevice);
    if (result!=cudaSuccess) return result;
    vectorDestroy(vector);
    vector=&cudaVector;
    return cudaSuccess;
}

cudaError_t cudaVectorUnload(Vector* vector) {
    Vector heapVector;
    heapVector.size=vector->size;

    cudaError_t result;
    result=cudaMemcpy(heapVector.dataArray,vector->dataArray,(heapVector.size*sizeof(double)),cudaMemcpyDeviceToHost);
    if (result!=cudaSuccess) return result;

    result=cudaFree(vector->dataArray);
    if (result!=cudaSuccess) return result;
    vector=&heapVector;
    return cudaSuccess;
}

cudaError_t cudaMatrixLoad(Matrix* matrix) {
    Matrix cudaMatrix;
}

cudaError_t cudaMatrixUnload(Matrix* matrix) {

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
    
    matrixConvertCSR(&matrix);
    
    printf("generating a vector of double for the moltiplication... \n");
    Vector vector;
    vectorCreate(&vector, matrix.colSize);

    printf("copying data in the CUDA memory... \n");
    cudaError_t cudaResult;
    cudaResult=cudaVectorLoad(&vector);
    if (cudaResult!=cudaSuccess) {
        printf("Error during data transfer phase: %d \n",cudaResult);
        return DATA_TRANSFER_ERROR;
    }
    cudaResult=cudaVectorUnload(&vector);
    if (cudaResult!=cudaSuccess) {
        printf("Error during data transfer phase: %d \n",cudaResult);
        return DATA_TRANSFER_ERROR;
    }
    
    printf("Operation done. Cleaning heap...1n");
    vectorDestroy(&vector);
    matrixDestroy(&matrix);
    
    
    return 0;
}   