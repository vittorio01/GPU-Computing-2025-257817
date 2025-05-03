SRC_FOLDER=src
BIN_FOLDER=bin
DATA_FOLDER=../data 

SRC_MAIN=main.c
OUT_MAIN=smm_csr_sequential

LIB_FOLDER=../libraries/

CMP=nvcc

SMM_CSR_SEQUENTIAL: MAKE_LIBS $(SRC_FOLDER)/$(SRC_MAIN)
	mkdir -p $(BIN_FOLDER)
	$(CMP) -o $(BIN_FOLDER)/$(OUT_MAIN) $(SRC_FOLDER)/$(SRC_MAIN) $(LIB_OBJ_FOLDER)/*.o -I $(LIB_HEADERS_FOLDER) -lm -g

clean: CLEAR_LIBS
	rm -rf $(BIN_FOLDER)

include $(LIB_FOLDER)/makefile