#include "COO_VM_multiplication.h"

void vmcoo_mul(Results *res,int* rowArray, int* colArray, double* dataArray, double* mulArray, int nnDataNumber, int mxColSize, int mxRowSize, int mxNNDataLenght) {
    res->array=(double*) malloc (sizeof(double) * mxRowSize);
    for (int i=0;i<mxRowSize;i++) {
        res->array[i]=0.0;
    }
    for (int i=0;i<mxNNDataLenght;i++) {
        res->array[(rowArray[i]-1)]+=(mulArray[colArray[i]-1]*dataArray[i]);
    }
    res->flops=mxRowSize+mxNNDataLenght;
}