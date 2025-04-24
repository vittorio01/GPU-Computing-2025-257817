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

void matrixDestroy(SparseMatrix* matrix) {
    free(matrix->colArray);
    free(matrix->dataArray);
    free(matrix->rowArray);
    matrix->colArray=NULL;
    matrix->dataArray=NULL;
    matrix->rowArray=NULL;
}

void vectorCreate(Vector* vector, int size) {
    vector->dataArray= (double*) malloc (sizeof(double)*size);
    srand(time(NULL));
    for (int i=0;i<size;i++) {
        vector->dataArray[i]=(double) rand() / RAND_MAX;
    }
}

void vectorDestroy(Vector* vector) {
    free(vector->dataArray);
    vector->dataArray=NULL;
}