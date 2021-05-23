.DEFAULT_GOAL = cli

LIBS = 

CC = g++
OUT = bin/ml-cli

INC = include 
CFLAGS = -Wall -g -I include -I lib/eigen -std=c++11	

DIRS = loaders cli classifiers optimizers

SRC = $(foreach dir,$(DIRS),$(wildcard src/$(dir)/*.cpp))
DEP = $(foreach dir,$(DIRS),$(wildcard include/$(dir)/*.h))
OBJ = $(SRC:.cpp=.o)

%.o: %.cpp $(DEP)
	$(CC) -c -o $@ $< $(CFLAGS)

cli: $(OBJ) $(DEP)
	$(CC) $(CFLAGS) -o $(OUT) $(LIBS) $(OBJ) 

.PHONY: clean
clean:
	rm $(OBJ) $(OUT)
