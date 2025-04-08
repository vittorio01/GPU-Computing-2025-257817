#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>
#include <time.h>

#include "COO_VM_multiplication/COO_VM_multiplication.h"
#include "math_stats/math_stats.h"

#define MISSING_MATRIX_INPUT_ERROR 1
#define FILE_OPEN_ERROR 2
#define DATA_FORMAT_ERROR 3

#define WARMUP_CYCLES 2
#define ITERATIONS 20


int main(int argc, char** argv) {
    if (argc < 1) {
        printf("Missing input matrix. Closing program...\n");
        return MISSING_MATRIX_INPUT_ERROR;
    }
    char* fileName=argv[1];
    FILE *filePointer=fopen(fileName,"r");
    if (filePointer==NULL) {
        printf("Cannot open file %s. Closing program...\n",fileName);
        return FILE_OPEN_ERROR;
    }

    char fileBuffer=fgetc(filePointer);
    while ((int)(fileBuffer) != EOF) { 
        if (fileBuffer==' ') {
            while ((int)(fileBuffer)!=EOF && fileBuffer!=' ') {
                fileBuffer=fgetc(filePointer);
            }
        }
        if (fileBuffer=='%') {
            while ((int)(fileBuffer)!=EOF && fileBuffer!='\n') {
                fileBuffer=fgetc(filePointer);
            }
        } else {
            break;
        }
        fileBuffer=fgetc(filePointer);
    }
    fseek(filePointer, -1, SEEK_CUR);
    int rowSize = 0;
    int colSize = 0;
    int dataSize = 0;
    if (fscanf(filePointer,"%d",&rowSize)==EOF) {
        printf("Data format error: row array size not valid\n");
        fclose(filePointer);
        return DATA_FORMAT_ERROR;
    }
    if (fscanf(filePointer,"%d",&colSize)==EOF) {
        printf("Data format error: column array size not valid\n");
        fclose(filePointer);
        return DATA_FORMAT_ERROR;
    }
    if (fscanf(filePointer,"%d",&dataSize)==EOF || (rowSize*colSize)<dataSize) {
        printf("Data format error: data array size not valid\n");
        fclose(filePointer);
        return DATA_FORMAT_ERROR;
    }
    if (rowSize==0 || colSize ==0 || dataSize == 0) return 0; 

    printf("Received matrix dimension: %dx%d with %d non-null terms\n",rowSize,colSize,dataSize);

    int *rowArray = (int*) malloc (sizeof(int)*dataSize);
    int *colArray = (int*) malloc (sizeof(int)*dataSize);
    double *dataArray = (double*) malloc (sizeof(double)*dataSize);
    
    for (int i=0;i<dataSize;i++) {
        if (fscanf(filePointer,"%d",&colArray[i])==EOF) {
            printf("Data format error: not enough elements\n");
            free(rowArray);
            free(colArray);
            free(dataArray);
            fclose(filePointer);
        }
        if (fscanf(filePointer,"%d",&rowArray[i])==EOF) {
            printf("Data format error: not enough elements\n");
            free(rowArray);
            free(colArray);
            free(dataArray);
            fclose(filePointer);
        }
        if (fscanf(filePointer,"%lf",&dataArray[i])==EOF) {
            printf("Data format error: not enough elements\n");
            free(rowArray);
            free(colArray);
            free(dataArray);
            fclose(filePointer);
        }
    }
    fclose(filePointer);

    double* mulArray=(double*) malloc(sizeof(double)*colSize);
    srand(time(NULL));
    printf("generating a vector of double for the moltiplication: \n");
    for (int i=0;i<colSize;i++) {
        mulArray[i]=(double) rand() / RAND_MAX;
        //printf("[%d] (%2f)\n",i,mulArray[i]);
    }
    printf("performing sparse matrix COO to vector multiplication... \n");

    struct timeval tv;
    double times[ITERATIONS];
    Results res;

    for (int i=-WARMUP_CYCLES;i<ITERATIONS;i++) {
        gettimeofday(&tv, NULL);
        times[i]=tv.tv_usec;
        vmcoo_mul(&res,rowArray,colArray,dataArray,mulArray,dataSize,colSize,rowSize,dataSize);
        gettimeofday(&tv, NULL);
        times[i] = tv.tv_usec - times[i]; 
    }
    double mean_value = math_geometric_mean(ITERATIONS,times);
    double variance = math_variance(ITERATIONS,times,mean_value);
    printf("Executed %d iterations, average time: %2f ms variance: %2f ms\n",ITERATIONS,(mean_value*pow(10,-3)),(variance*pow(10,-3)));
    double flops= (double)(((double)(res.flops)/mean_value)*pow(10,-3));
    printf("average Giga FLOP/s: %2f\n",flops);
 
    printf("Clearing heap...\n");
    free(res.array);
    free(rowArray);
    free(colArray);
    free(dataArray);
    free(mulArray);
    
    return 0;
}   