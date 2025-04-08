SRC_FOLDER=src
BIN_FOLDER=bin
DATA_FOLDER=../data 

SRC_MAIN=main.c
OUT_MAIN=smm_coo_sequential

LIB_FOLDER=../libraries/

SMM_COO_SEQUENTIAL: MAKE_LIBS $(SRC_FOLDER)/$(SRC_MAIN)
	mkdir -p $(BIN_FOLDER)
	gcc -o $(BIN_FOLDER)/$(OUT_MAIN) $(SRC_FOLDER)/$(SRC_MAIN) -lm $(LIB_OBJ_FOLDER)/*.o -I $(LIB_HEADERS_FOLDER)

clean: CLEAR_LIBS
	rm -rf $(BIN_FOLDER)

include $(LIB_FOLDER)/makefile