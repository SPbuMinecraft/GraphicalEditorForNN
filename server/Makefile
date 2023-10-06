# USAGE:
# cd server
# make
# ./test
# make clean

# Compiler
CC = g++
# Compiler flags
FLAGS = -g -Wall -std=c++17

SRC=Blob.cpp Operation.cpp Tensor.cpp main.cpp
OBJ=Blob.o Operation.o Tensor.o main.o

TARGET=main

all: $(TARGET)

# With all the object files in place, compile the final target
$(TARGET): $(OBJ)
	$(CC) $(FLAGS) $(OBJ) -o $@

# Compile all source files with $(HDR) directory included in search path
%.o: %.cpp
	$(CC) $(FLAGS) -c $< -o $@

test: BlobTests.o Blob.o
	$(CC) $(FLAGS) $^ -o $@

clean:
	rm main test $(OBJ) BlobTests.o
