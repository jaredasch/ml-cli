.DEFAULT_GOAL = cli

LIBS = -llapack -lblas -larmadillo

CC = g++
OUT = bin/ml-cli

INC = include
CFLAGS = -Wall -g -I $(INC)

SRC = $(wildcard src/loaders/*.cpp) $(wildcard src/cli/*.cpp)
DEP = $(wildcard src/loaders/*.h) $(wildcard src/cli/*.h)
OBJ = $(SRC:.cpp=.o)

%.o: %.cpp $(DEP)
	$(CC) -c -o $@ $< $(CFLAGS)

cli: $(OBJ)
	echo $(SRC)
	$(CC) $(CFLAGS) -o $(OUT) $(LIBS) $(OBJ) 

.PHONY: clean
clean:
	rm $(OBJ) $(OUT)
