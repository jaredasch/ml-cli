.DEFAULT_GOAL = cli

LIBS = -llapack -lblas -larmadillo

CC = g++
OUT = bin/ml-cli

INC = include
CFLAGS = -Wall -g -I $(INC) -std=c++11	

DIRS = loaders cli classifiers optimizers

SRC = $(foreach dir,$(DIRS),$(wildcard src/$(dir)/*.cpp))
DEP = $(foreach dir,$(DIRS),$(wildcard src/$(dir)/*.h))
OBJ = $(SRC:.cpp=.o)

%.o: %.cpp $(DEP)
	$(CC) -c -o $@ $< $(CFLAGS)

cli: $(OBJ) $(DEP)
	$(CC) $(CFLAGS) -o $(OUT) $(LIBS) $(OBJ) 

.PHONY: clean
clean:
	rm $(OBJ) $(OUT)
