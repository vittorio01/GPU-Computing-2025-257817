#ifndef __COO_VM_MUL_H__
#define __COO_VM_MUL_H__

#include <stdio.h>
#include <stdlib.h>


typedef struct Results {
    double* array;
    int flops;
} Results;

void vmcoo_mul(Results *res, int* rowArray, int* colArray, double* dataArray, double* mulArray, int nnDataNumber, int mxColSize, int mxRowSize,int mxNNDataLenght);

#endif