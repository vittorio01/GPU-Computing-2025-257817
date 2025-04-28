#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>

#include "math.h"
#include "dataLib.h"
#include "mathStats.h"

#define MISSING_MATRIX_INPUT_ERROR 10
#define DATA_TRANSFER_ERROR 11



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
        printf("Error during vector data transfer phase: %s \n",cudaGetErrorString(cudaResult));
        return DATA_TRANSFER_ERROR;
    }
    cudaResult=cudaMatrixLoad(&matrix);
    if (cudaResult!=cudaSuccess) {
        printf("Error during matrix data transfer phase: %s \n",cudaGetErrorString(cudaResult));
        return DATA_TRANSFER_ERROR;
    }
    printf("copying results to the RAM memory...\n");
    cudaResult=cudaMatrixUnload(&matrix);
    if (cudaResult!=cudaSuccess) {
        printf("Error during vector data transfer phase: %s \n",cudaGetErrorString(cudaResult));
        return DATA_TRANSFER_ERROR;
    }
    
    printf("Operation done. Cleaning heap...1n");
    vectorDestroy(&vector);
    matrixDestroy(&matrix);
    
    
    return 0;
}   