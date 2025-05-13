//Library used for defining Sparse Matrix and Vectors structures and relative functions

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

/*
------ Sparse Matrix structure ------
This structure contains the definition of a Sparse Matrix:
- type -> COO or CSR, the rype of the matrix
- rowSize -> the rows number of the matrix
- colSize -> the columns number of the matrix
- notNull -> the number of not null elements

- rowArray -> the pointer of an int array that contains all rows coefficients in a CSR or COO format for not null elements
- colArray -> the pointer of an int array that contains all column coefficients for not null elements
- nutNull  -> the pointer of a float array that contains the value of all not null elements

*/
typedef enum {COO,CSR} sparseMatrixType;
typedef struct SparseMatrix {
    int rowSize;
    int colSize;
    int notNull;
    sparseMatrixType type;

    int* rowArray;
    int* colArray;
    float* dataArray;

} SparseMatrix;

/*
------- Vector structure -------
The structure that contains the definition of a Vector:
- size      -> the size of the vector
- dataArray -> A pointer for the array of floats that defines the vector
*/
typedef struct Vector {
    int size;

    float* dataArray;

} Vector;



//------ Functions ------

/*matrixOpen is a function that loads data from a file to a SparseMatrix structure:
- filePath -> the String that contains the name of the file to read 
- matrix -> an empty SparseMatrix structure where data should be loaded

  returns -> the result of the operation
*/
int matrixOpen(char* filePath, SparseMatrix* matrix);

/*matrixDestroy is a function that unload the arrays in a SparseMatrix structure:
- matrix -> SparseMatrix structure to clear.

  returns -> the result of the operation (cudaError_t)
*/
cudaError_t matrixDestroy(SparseMatrix* matrix);

/*matriConvertCSR is a function that converts the row array of a SparseMatrix structure from the COO to the CSR format: 
- matrix -> SparseMatrix structure to convert.

  returns -> the result of the operation (cudaError_t)
*/
cudaError_t matrixConvertCSR(SparseMatrix* matrix);

/*vectorCreate is a function that assign values to a Vector structure and initializes it to 0:
- vector -> the vector to initialize
- size -> the size of the vector

  returns -> the result of the operation (cudaError_t)
*/
cudaError_t vectorCreate(Vector* vector, int size);

/*vectorCreateRandom is a function that assign values to a Vector structure and initializes it with random values:
- vector -> the vector to initialize
- size -> the size of the vector

  returns -> the result of the operation (cudaError_t)
*/
cudaError_t vectorCreateRandom(Vector* vector, int size);

/*vectorDestroy is a function that unload the array in a Vector structure:
- vector -> SparseMatrix structure to clear.

  returns -> the result of the operation (cudaError_t)
*/
cudaError_t vectorDestroy(Vector* vector);

/*vectorPrefetch is a function that loads the array in the Vector structure in the GPU global memory:
- vector -> Vector structure to load.

  returns -> the result of the operation (cudaError_t)
*/
cudaError_t vectorPrefetch(Vector* vector,int cudaDevice);

/*matrixPrefetch is a function that loads the vectors in the SparseMatrix structure in the GPU global memory:
- matrix -> SparseMatrix structure to load.

  returns -> the result of the operation (cudaError_t)
*/
cudaError_t matrixPrefetch(SparseMatrix* matrix,int cudaDevice);

#ifdef __cplusplus
}
#endif

#endif 