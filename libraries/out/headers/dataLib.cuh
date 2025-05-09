#ifndef __DATA_LIB_H__
#define __DATA_LIB_H__

#define FILE_OPEN_ERROR 1
#define ROW_ARRAY_NOT_VALID 2
#define COL_ARRAY_NOT_VALID 3
#define DATA_ARRAY_NOT_VALID 4
#define FILE_TRUNCATED 5

#define MATRIX_READED 0

#include "stdio.h"
#include "stdlib.h"
#include <time.h>

typedef enum {COO,CSR} sparseMatrixType;
typedef enum {DEVICE,HOST,NONE} objectAllocation;

typedef struct SparseMatrix {
    int* rowArray;
    int* colArray;
    float* dataArray;

    int rowSize;
    int colSize;
    int notNull;

    sparseMatrixType type;
    objectAllocation pos;

} SparseMatrix;

typedef struct Vector {
    float* dataArray;
    int size;

    objectAllocation pos;
} Vector;

int matrixOpen(char* filePath, SparseMatrix* matrix);
void matrixDestroy(SparseMatrix* matrix);

void matrixConvertCSR(SparseMatrix* matrix);

void vectorCreate(Vector* vector, int size);
void vectorDestroy(Vector* vector);

cudaError_t cudaMatrixUnload(SparseMatrix* matrix);
cudaError_t cudaMatrixLoad(SparseMatrix* matrix);
cudaError_t cudaVectorLoad(Vector* vector);
cudaError_t cudaVectorUnload(Vector* vector);

#endif 