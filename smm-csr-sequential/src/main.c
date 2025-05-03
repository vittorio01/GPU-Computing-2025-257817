#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>

#include "math.h"
#include "dataLib.h"
#include "mathStats.h"

#define MISSING_MATRIX_INPUT_ERROR 10

#define WARMUP_CYCLES 2
#define ITERATIONS 10


void vmcsr_mul(Vector* output, int* floats, SparseMatrix* matrix,Vector* vector) {
    output->size=matrix->rowSize;  

    for (int i=0;i<output->size;i++) output->dataArray[i]=0;
    for (int i=0;i<(matrix->rowSize);i++) {
        for (int j=(matrix->rowArray[i]);j<(matrix->rowArray[i+1]);j++) {
            output->dataArray[i]+=matrix->dataArray[j]*vector->dataArray[matrix->colArray[j]];
        }
    }
    *floats=matrix->notNull+output->size;
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

    
    printf("performing sparse matrix CSR to vector multiplication... \n");

    struct timeval tv;
    double times[ITERATIONS];
    Vector output;
    vectorCreate(&output,matrix.rowSize);
    int floats=0;
    for (int i=-WARMUP_CYCLES;i<ITERATIONS;i++) {
        gettimeofday(&tv, NULL);
        times[i]=tv.tv_usec;
        printf("iteration %d took %2f\n",i,times[i]);
        vmcsr_mul(&output,&floats,&matrix,&vector);
        gettimeofday(&tv, NULL);
        times[i] = (tv.tv_usec - times[i]); 
    }
    double mean_value = math_geometric_mean(ITERATIONS,times);
    double variance = math_variance(ITERATIONS,times,mean_value);
    printf("Executed %d iterations, floating point operations: %d average time: %2f micros variance: %2f micros\n",ITERATIONS,floats,(mean_value),(variance));
    double flops= (double)(((double)(floats)/(mean_value)));
    printf("average Giga FLOP/s: %2f\n",flops);
    printf("Clearing heap...\n");
    vectorDestroy(&vector);
    matrixDestroy(&matrix);
    
    
    return 0;
}   