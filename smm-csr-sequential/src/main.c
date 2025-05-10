#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>

#include "math.h"
#include "dataLib.h"
#include "mathStats.h"

#define MISSING_MATRIX_INPUT_ERROR 10

#define WARMUP_CYCLES 5
#define ITERATIONS 20


void vmcsr_mul(Vector* output, SparseMatrix* matrix,Vector* vector) {
    output->size=matrix->rowSize;  
    for (int i=0;i<(matrix->rowSize);i++) {
        int acc=0;
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
    
    printf("generating a vector of float for the moltiplication... \n");
    Vector vector;
    vectorCreate(&vector, matrix.colSize);

    
    printf("performing sparse matrix CSR to vector multiplication... \n");

    struct timeval tv;
    float times[ITERATIONS];
    Vector output;
    vectorCreate(&output,matrix.rowSize);
    for (int i=-WARMUP_CYCLES;i<ITERATIONS;i++) {
        gettimeofday(&tv, NULL);
        if (i>=0) times[i]=tv.tv_usec;
        vmcsr_mul(&output,&matrix,&vector);
        gettimeofday(&tv, NULL);
        if (i>=0) {
            times[i] = (tv.tv_usec - times[i])*pow(10,-3);
            printf("iteration %d took %f ms\n",i,times[i]);
        }
    }
    double mean_value = math_geometric_mean(ITERATIONS,times);
    double variance = math_variance(ITERATIONS,times,mean_value);
    int floats=2*matrix.notNull;
    printf("Executed %d iterations, floating point operations: %d average time: %2f ms variance: %2f ms\n",ITERATIONS,matrix.notNull,(mean_value),(variance));
    double flops= ((double)floats)/(mean_value*pow(10,-3));
    printf("average GFLOP/s: %2f\n",flops*pow(10,-9));

    printf("Clearing heap...\n");
    vectorDestroy(&vector);
    vectorDestroy(&output);
    matrixDestroy(&matrix);
    
    
    return 0;
}   