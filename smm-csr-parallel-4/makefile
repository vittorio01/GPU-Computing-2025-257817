SRC_FOLDER=src
BIN_FOLDER=bin
OUT_FOLDER=out
DATA_FOLDER=../data 

SRC_MAIN=main.cu
OUT_MAIN=smm-csr-parallel-4

LIB_FOLDER=../libraries/

CMP=nvcc

SMM-CSR-PARALLEL-4: MAKE_LIBS $(SRC_FOLDER)/$(SRC_MAIN)
	mkdir -p $(BIN_FOLDER)
	mkdir -p $(OUT_FOLDER)
	$(CMP) -o $(BIN_FOLDER)/$(OUT_MAIN) $(SRC_FOLDER)/$(SRC_MAIN) -lm $(LIB_OBJ_FOLDER)/*.o -I $(LIB_HEADERS_FOLDER)

clean: CLEAR_LIBS
	rm -rf $(BIN_FOLDER)
	rm -rf $(OUT_FOLDER)

include $(LIB_FOLDER)/makefile