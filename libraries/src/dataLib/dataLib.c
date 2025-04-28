#include "dataLib.h"

int matrixOpen(char* filePath, SparseMatrix* matrix) {
    FILE *filePointer=fopen(filePath,"r");
    if (filePointer==NULL) {
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
    if (fscanf(filePointer,"%d",&matrix->rowSize)==EOF) {
        fclose(filePointer);
        return ROW_ARRAY_NOT_VALID;
    }
    if (fscanf(filePointer,"%d",&matrix->colSize)==EOF) {
        fclose(filePointer);
        return COL_ARRAY_NOT_VALID;
    }
    if (fscanf(filePointer,"%d",&matrix->notNull)==EOF || (matrix->rowSize*matrix->colSize)<matrix->notNull) {
        fclose(filePointer);
        return DATA_ARRAY_NOT_VALID;
    }
    if (matrix->rowSize==0 || matrix->colSize ==0 || matrix->notNull == 0) return 0; 

    matrix->rowArray = (int*) malloc (sizeof(int)*matrix->notNull);
    matrix->colArray = (int*) malloc (sizeof(int)*matrix->notNull);
    matrix->dataArray = (double*) malloc (sizeof(double)*matrix->notNull);
    matrix->type=COO;
    matrix->pos=HOST;
    for (int i=0;i<(matrix->notNull);i++) {
        if (fscanf(filePointer,"%d",&matrix->colArray[i])==EOF) {
            matrixDestroy(matrix);
            fclose(filePointer);
            return FILE_TRUNCATED;
        }
        if (fscanf(filePointer,"%d",&matrix->rowArray[i])==EOF) {
            matrixDestroy(matrix);
            fclose(filePointer);
            return FILE_TRUNCATED;
        }
        if (fscanf(filePointer,"%lf",&matrix->dataArray[i])==EOF) {
            matrixDestroy(matrix);
            fclose(filePointer);
            return FILE_TRUNCATED;
        }
        matrix->colArray[i]--;
        matrix->rowArray[i]--;
    }
    fclose(filePointer);
}

void matrixConvertCSR(SparseMatrix* matrix) {
    if (matrix->type==CSR) return;
    printf("converting to CSR...\n");
    matrix->type=CSR;
    int *newRow= (int*) malloc (sizeof(int) * (matrix->rowSize+1));
    for (int i=0;i<matrix->rowSize+1;i++) {
        newRow[i]=0;
    }
   
    for (int rowIndex=0;rowIndex<matrix->notNull;rowIndex++) {
        newRow[(matrix->rowArray[rowIndex])+1]++;
    }
   
    for (int i = 0; i < matrix->rowSize; i++) {
        newRow[i + 1] += newRow[i];
    }
    free(matrix->rowArray);
    matrix->rowArray=newRow;
}

cudaError_t matrixDestroy(SparseMatrix* matrix) {
    cudaError_t result;
    if (matrix->pos==HOST) {
        free(matrix->colArray);
        free(matrix->dataArray);
        free(matrix->rowArray);
    } else {
        result=cudaFree(matrix->rowArray);
        if (result!=cudaSuccess) return result;
        result=cudaFree(matrix->dataArray);
        if (result!=cudaSuccess) return result;
        result=cudaFree(matrix->colArray);
        if (result!=cudaSuccess) return result;
    }
    matrix->colArray=NULL;
    matrix->dataArray=NULL;
    matrix->rowArray=NULL;
    matrix->colSize=0;
    matrix->rowSize=0;
    matrix->notNull=0;
    matrix->pos=NONE;
    return cudaSuccess;
}

void vectorCreate(Vector* vector, int size) {
    vector->dataArray= (double*) malloc (sizeof(double)*size);
    vector->size=size;
    vector->pos=HOST;
    srand(time(NULL));
    for (int i=0;i<size;i++) {
        vector->dataArray[i]=(double) rand() / RAND_MAX;
    }
}

cudaError_t vectorDestroy(Vector* vector) {
    cudaError_t result;
    switch (vector->pos) {
        case HOST:
            free(vector->dataArray);
            break;
        
        case DEVICE:
            cudaFree(vector->dataArray);
            break;
    }
    vector->size=0;
    vector->dataArray=NULL;
    vector->pos=NONE;
}

cudaError_t cudaVectorLoad(Vector* vector) {
    double* vectorData=vector->dataArray;
    double* cudaData=NULL;

    cudaError_t result;
    result=cudaMalloc((void**)&cudaData,(vector->size*sizeof(double)));
    if (result!=cudaSuccess) return result;

    result=cudaMemcpy(cudaData,vectorData,(vector->size*sizeof(double)),cudaMemcpyHostToDevice);
    if (result!=cudaSuccess) return result;
    free(vectorData);
    vector->dataArray=cudaData;
    vector->pos=DEVICE;
    return cudaSuccess;
}

cudaError_t cudaVectorUnload(Vector* vector) {
    double* cudaData=vector->dataArray;
    double* heapData=(double*) malloc (vector->size*sizeof(double));

    cudaError_t result;
    
    result=cudaMemcpy(heapData,cudaData,(vector->size*sizeof(double)),cudaMemcpyDeviceToHost);
    if (result!=cudaSuccess) return result;
    
    result=cudaFree(cudaData);
    if (result!=cudaSuccess) return result;
    vector->dataArray=heapData;
    vector->pos=HOST;
    return cudaSuccess;
}


cudaError_t cudaMatrixLoad(SparseMatrix* matrix) {
    int* matrixColArray=matrix->colArray;
    int* matrixRowArray=matrix->rowArray;
    double* matrixDataArray=matrix->dataArray;

    int* cudaColArray=NULL;
    int* cudaRowArray=NULL;
    double* cudaDataArray=NULL;

    cudaError_t result;
    result=cudaMalloc((void**)&cudaColArray,(matrix->notNull*sizeof(int)));
    if (result!=cudaSuccess) return result;
    result=cudaMemcpy(cudaColArray,matrixColArray,(matrix->notNull*sizeof(int)),cudaMemcpyHostToDevice);
    if (result!=cudaSuccess) return result;

    result=cudaMalloc((void**)&cudaDataArray,(matrix->notNull*sizeof(double)));
    if (result!=cudaSuccess) return result;
    result=cudaMemcpy(cudaDataArray,matrixDataArray,(matrix->notNull*sizeof(double)),cudaMemcpyHostToDevice);
    if (result!=cudaSuccess) return result;

    if (matrix->type==CSR) {
        result=cudaMalloc((void**)&cudaRowArray,((matrix->rowSize)+1)*sizeof(int));
        if (result!=cudaSuccess) return result;

        result=cudaMemcpy(cudaRowArray,matrixRowArray,(matrix->notNull*sizeof(int)),cudaMemcpyHostToDevice);
    } else {
        result=cudaMalloc((void**)&cudaRowArray,(matrix->notNull)*sizeof(int));
        if (result!=cudaSuccess) return result;
        
        result=cudaMemcpy(cudaRowArray,matrixRowArray,((matrix->notNull)*sizeof(int)),cudaMemcpyHostToDevice);
    }
    if (result!=cudaSuccess) return result;

    free(matrixColArray);
    free(matrixRowArray);
    free(matrixDataArray);
    matrix->colArray=cudaColArray;
    matrix->rowArray=cudaRowArray;
    matrix->dataArray=cudaDataArray;
    matrix->pos=DEVICE;
    return cudaSuccess;
}

cudaError_t cudaMatrixUnload(SparseMatrix* matrix) {
    int* cudaColArray=matrix->colArray;
    int* cudaRowArray=matrix->rowArray;
    double* cudaDataArray=matrix->dataArray;

    int* heapColArray=(int*) malloc (matrix->notNull *sizeof(int));
    double* heapDataArray=(double*) malloc (matrix->notNull *sizeof(double));
    int* heapRowArray=NULL;
    if (matrix->type==CSR) {
        heapRowArray=(int*) malloc (((matrix->rowSize)+1) *sizeof(int));
    } else {
        heapRowArray=(int*) malloc (matrix->notNull *sizeof(int));
    }

    cudaError_t result;
    result=cudaMemcpy(heapColArray,cudaColArray,(matrix->notNull*sizeof(int)),cudaMemcpyDeviceToHost);
    if (result!=cudaSuccess) return result;
    result=cudaMemcpy(heapDataArray,cudaDataArray,(matrix->notNull*sizeof(double)),cudaMemcpyDeviceToHost);
    if (result!=cudaSuccess) return result;

    if (matrix->type==CSR) {
        result=cudaMemcpy(heapRowArray,cudaRowArray,(matrix->notNull*sizeof(int)),cudaMemcpyDeviceToHost);
    } else {
        
        result=cudaMemcpy(heapRowArray,cudaRowArray,((matrix->notNull)*sizeof(int)),cudaMemcpyDeviceToHost);
    }
    if (result!=cudaSuccess) return result;
    result=cudaFree(cudaRowArray);
    if (result!=cudaSuccess) return result;
    result=cudaFree(cudaColArray);
    if (result!=cudaSuccess) return result;
    result=cudaFree(cudaDataArray);
    if (result!=cudaSuccess) return result;
    matrix->colArray=heapColArray;
    matrix->rowArray=heapRowArray;
    matrix->dataArray=heapDataArray;
    matrix->pos=HOST;
    return cudaSuccess;
}
