#ifndef __DATA_LIB_H__
#define __DATA_LIB_H__

#define FILE_OPEN_ERROR 1
#define ROW_ARRAY_NOT_VALID 2
#define COL_ARRAY_NOT_VALID 3
#define DATA_ARRAY_NOT_VALID 4
#define FILE_TRUNCATED 5
#define MEMORY_ALLOCATION_ERROR 6
#define MATRIX_READED 0

#ifdef __cplusplus
extern "C" {
#endif

#include "stdio.h"
#include "stdlib.h"
#include <time.h>
#include <cuda_runtime.h>

typedef enum {COO,CSR} sparseMatrixType;

typedef struct SparseMatrix {
    int rowSize;
    int colSize;
    int notNull;

    sparseMatrixType type;

    int* rowArray;
    int* colArray;
    double* dataArray;

} SparseMatrix;

typedef struct Vector {
    int size;

    double* dataArray;

} Vector;

int matrixOpen(char* filePath, SparseMatrix* matrix);
cudaError_t matrixDestroy(SparseMatrix* matrix);

cudaError_t matrixConvertCSR(SparseMatrix* matrix);

cudaError_t vectorCreate(Vector* vector, int size);
cudaError_t vectorCreateRandom(Vector* vector, int size);
cudaError_t vectorDestroy(Vector* vector);

cudaError_t vectorPrefetch(Vector* vector,int cudaDevice);
cudaError_t matrixPrefetch(SparseMatrix* matrix,int cudaDevice);

#ifdef __cplusplus
}
#endif

#endif 