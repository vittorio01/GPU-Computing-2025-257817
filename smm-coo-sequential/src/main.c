#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>

#include "math.h"
#include "dataLib.h"
#include "mathStats.h"

#define MISSING_MATRIX_INPUT_ERROR 10

#define WARMUP_CYCLES 2
#define ITERATIONS 20

typedef struct Results {
    Vector* output;
    int floats;
} Results;

void vmcoo_mul(Results *res,SparseMatrix* matrix,Vector* vector) {
    res->output->dataArray=(double*) malloc (sizeof(double) * matrix->rowSize);
    for (int i=0;i<matrix->rowSize;i++) {
        res->output->dataArray[i]=0.0;
    }
    for (int i=0;i<matrix->notNull;i++) {
        res->output->dataArray[(matrix->rowArray[i]-1)]+=(vector->dataArray[matrix->colArray[i]-1]*matrix->dataArray[i]);
    }
    res->floats=matrix->rowSize+matrix->notNull;
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
    
    
    printf("generating a vector of double for the moltiplication... \n");
    Vector vector;
    vectorCreate(&vector, matrix.colSize);

    
    printf("performing sparse matrix COO to vector multiplication... \n");

    struct timeval tv;
    double times[ITERATIONS];
    Results res;

    for (int i=-WARMUP_CYCLES;i<ITERATIONS;i++) {
        gettimeofday(&tv, NULL);
        times[i]=tv.tv_usec;
        vmcoo_mul(&res,&matrix,&vector);
        gettimeofday(&tv, NULL);
        times[i] = tv.tv_usec - times[i]; 
    }
    double mean_value = math_geometric_mean(ITERATIONS,times);
    double variance = math_variance(ITERATIONS,times,mean_value);
    printf("Executed %d iterations, average time: %2f ms variance: %2f ms\n",ITERATIONS,(mean_value*pow(10,-3)),(variance*pow(10,-3)));
    double flops= (double)(((double)(res.floats)/mean_value)*pow(10,-3));
    printf("average Giga FLOP/s: %2f\n",flops);

    printf("Clearing heap...\n");
    vectorDestroy(&vector);
    matrixDestroy(&matrix);
    
    
    return 0;
}   