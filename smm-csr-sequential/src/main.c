/*Implementation of the sequential SpMV multiplication */


#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>

#include "math.h"
#include "dataLib.h"
#include "mathStats.h"

#define MISSING_MATRIX_INPUT_ERROR 10

#define WARMUP_CYCLES 5
#define ITERATIONS 20


/*
Implementation of the sequential SpMV moltiplication for CSR matrices. 
A for cycle scans the row vector and decodes the starting pointer and the ending pointer to column and data arrays. 

Then a second for cycle iterates on the subspace from startRow to endRow to multiply all terms in a matrix row with the correspondent vector element 
and accumulates the result into the variable acc. 

The result is then saved in the output vector.
*/

void vmcsr_mul(Vector* output, SparseMatrix* matrix,Vector* vector) {
    output->size=matrix->rowSize;  
    for (int i=0;i<(matrix->rowSize);i++) {
        int startRow=matrix->rowArray[i];
        int endRow=matrix->rowArray[i+1];
        int acc=0;
        for (int j=startRow;j<endRow;j++) {
            acc+=matrix->dataArray[j]*vector->dataArray[matrix->colArray[j]];
        }
        
        output->dataArray[i]=acc;
    }
}

int main(int argc, char** argv) {
    //Data loading phase: the programs checks if there is a matrix, opens it using the datalib custom library and creates a vector for the multiplication
    if (argc < 2) {
        printf("Missing input matrix-> Closing program...\n");
        return MISSING_MATRIX_INPUT_ERROR;
    }

    //Creating matrix structure and opening the file
    SparseMatrix* matrix;
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

        case DATA_ARRAY_NOT_VALID:
        printf("Error during matrix reading phase: data array not valid\n");
        return DATA_ARRAY_NOT_VALID;

        case MEMORY_ALLOCATION_ERROR:
        printf("Error during matrix reading phase: error during memory allocation\n");
        return MEMORY_ALLOCATION_ERROR;

        case FILE_TRUNCATED:
        printf("Error during matrix reading phase: cannot open file\n");
        return FILE_TRUNCATED;

        default:
        printf("Matrix read successfully :)\n");
        break;
    }
    
    matrixConvertCSR(matrix);
    
    
    //Creating input and output vectors
    printf("generating a vector of float for the moltiplication... \n");
    Vector* vector;
    vectorCreate(&vector, matrix->colSize);
    Vector* output;
    vectorCreate(&output,matrix->rowSize);
    
    printf("performing sparse matrix CSR to vector multiplication... \n");

    //Benchmarking phase: Using the function gettimeofday to measure the time used by the algorithm
    struct timeval tv;
    float times[ITERATIONS];
    for (int i=-WARMUP_CYCLES;i<ITERATIONS;i++) {
        gettimeofday(&tv, NULL);
        if (i>=0) times[i]=tv.tv_usec;
        vmcsr_mul(output,matrix,vector);
        gettimeofday(&tv, NULL);
        if (i>=0) {
            times[i] = (tv.tv_usec - times[i])*pow(10,-3);
            if (times[i]<=0) times[i]=times[i-1];
            printf("iteration %d took %f ms\n",i,times[i]);
        }
    }

    //Performing the average GFLOP/S
    double mean_value = math_geometric_mean(ITERATIONS,times);
    double variance = math_variance(ITERATIONS,times,mean_value);
    int floats=2*matrix->notNull;
    printf("Executed %d iterations, floating point operations: %d average time: %2f ms variance: %2f ms\n",ITERATIONS,matrix->notNull,(mean_value),(variance));
    double flops= ((double)floats)/(mean_value*pow(10,-3));
    printf("average GFLOP/s: %2f\n",flops*pow(10,-9));

    //Clearing structures
    printf("Clearing heap...\n");
    vectorDestroy(vector);
    vectorDestroy(output);
    matrixDestroy(matrix);
    
    
    return 0;
}   