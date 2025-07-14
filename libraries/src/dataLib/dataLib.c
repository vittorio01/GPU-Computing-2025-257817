#include "dataLib.h"

cudaError_t matrixCreate(SparseMatrix** matrix) {
    return cudaMallocManaged((void**)matrix, sizeof(SparseMatrix),cudaMemAttachHost);
}

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
    if (fscanf(filePointer,"%d",&matrix->colSize)==EOF) {
        fclose(filePointer);
        return COL_ARRAY_NOT_VALID;
    }
    if (fscanf(filePointer,"%d",&matrix->rowSize)==EOF) {
        fclose(filePointer);
        return ROW_ARRAY_NOT_VALID;
    }
    if (fscanf(filePointer,"%d",&matrix->notNull)==EOF || (matrix->rowSize*matrix->colSize)<matrix->notNull) {
        fclose(filePointer);
        return DATA_ARRAY_NOT_VALID;
    }
    if (matrix->rowSize==0 || matrix->colSize ==0 || matrix->notNull == 0) return 0; 

    cudaError_t cudaResult;
    cudaResult=cudaMallocManaged((void**)&matrix->rowArray, sizeof(int)*matrix->notNull,cudaMemAttachHost);
    if (cudaResult!=cudaSuccess) {
        fclose(filePointer);
        return MEMORY_ALLOCATION_ERROR;
    }
    cudaResult=cudaMallocManaged((void**)&matrix->colArray, sizeof(int)*matrix->notNull,cudaMemAttachHost);
    if (cudaResult!=cudaSuccess) {
        fclose(filePointer);
        return MEMORY_ALLOCATION_ERROR;
    }
    cudaResult=cudaMallocManaged((void**)&matrix->dataArray, sizeof(float)*matrix->notNull,cudaMemAttachHost);
    if (cudaResult!=cudaSuccess) {
        fclose(filePointer);
        return MEMORY_ALLOCATION_ERROR;
    }

    matrix->type=COO;
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
        if (fscanf(filePointer,"%f",&matrix->dataArray[i])==EOF) {
            matrixDestroy(matrix);
            fclose(filePointer);
            return FILE_TRUNCATED;
        }
        matrix->colArray[i]--;
        matrix->rowArray[i]--;
    }
    fclose(filePointer);
}

cudaError_t matrixConvertCSR(SparseMatrix* matrix) {
    if (matrix->type==CSR) return cudaSuccess;
    matrix->type=CSR;
    
    int *newRow=NULL;
    cudaError_t result=cudaMallocManaged((void**)&newRow, sizeof(int)*(matrix->rowSize+1),cudaMemAttachHost); 
    if (result!=cudaSuccess) return result;
    for (int i=0;i<matrix->rowSize+1;i++) {
        newRow[i]=0;
    }
   
    for (int rowIndex=0;rowIndex<matrix->notNull;rowIndex++) {
        newRow[(matrix->rowArray[rowIndex])+1]++;
    }
   
    for (int i = 0; i < matrix->rowSize; i++) {
        newRow[i + 1] += newRow[i];
    }
    result=cudaFree(matrix->rowArray);
    if (result!=cudaSuccess) return result;
    matrix->rowArray=newRow;
    return cudaSuccess;
}

cudaError_t matrixDestroy(SparseMatrix* matrix) {
    cudaError_t result;
    result=cudaFree(matrix->rowArray);
    if (result!=cudaSuccess) return result;
    result=cudaFree(matrix->dataArray);
    if (result!=cudaSuccess) return result;
    result=cudaFree(matrix->colArray);
    if (result!=cudaSuccess) return result;
    matrix->colArray=NULL;
    matrix->dataArray=NULL;
    matrix->rowArray=NULL;
    matrix->colSize=0;
    matrix->rowSize=0;
    matrix->notNull=0;
    result=cudaFree(matrix);
    return cudaSuccess;
}

cudaError_t vectorCreate(Vector** vector, int size) {
    cudaError_t result=cudaMallocManaged((void**)vector, sizeof(Vector),cudaMemAttachHost);
    if (result!=cudaSuccess) return result;
    (*vector)->dataArray=NULL;
    result=cudaMallocManaged((void**)&(*vector)->dataArray, sizeof(float)*size,cudaMemAttachHost);
    if (result!=cudaSuccess) return result;
    (*vector)->size=size;
    return cudaSuccess;
}

cudaError_t vectorCreateRandom(Vector** vector, int size) {
    cudaError_t result=vectorCreate(vector,size);
    if (result!=cudaSuccess) return result;
    srand(time(NULL));
    for (int i=0;i<size;i++) {
        (*vector)->dataArray[i]=1;//vector->dataArray[i]=(float) rand() / RAND_MAX;
    }
    return cudaSuccess;
}
cudaError_t vectorDestroy(Vector* vector) {
    cudaError_t result=cudaFree(vector->dataArray);
    if (result!=cudaSuccess) return result;
    vector->size=0;
    vector->dataArray=NULL;
    result=cudaFree(vector);
    return cudaSuccess;
}

cudaError_t vectorPrefetch(Vector* vector,int cudaDevice)  {
    return cudaMemPrefetchAsync((void*)vector->dataArray, vector->size*sizeof(float),cudaDevice,0);

}
cudaError_t matrixPrefetch(SparseMatrix* matrix,int cudaDevice) {
    cudaError_t result;
    result=cudaMemPrefetchAsync(matrix->colArray, (matrix->notNull)*sizeof(int),cudaDevice,0);
    if (result!=cudaSuccess) return result;
    result=cudaMemPrefetchAsync((void*)matrix->dataArray, (matrix->notNull)*sizeof(float),cudaDevice,0);
    if (result!=cudaSuccess) return result;
    if (matrix->type==CSR) {
        result=cudaMemPrefetchAsync((void*)matrix->rowArray, ((matrix->rowSize)+1)*sizeof(int),cudaDevice,0);
    } else {
        result=cudaMemPrefetchAsync((void*)matrix->rowArray, (matrix->notNull)*sizeof(int),cudaDevice,0);
    }
    return result;
}