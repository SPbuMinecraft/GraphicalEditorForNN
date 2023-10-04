# USAGE:
# cd server
# make
# ./test
# make clean

# Compiler
CC = g++
# Compiler flags
FLAGS = -g -Wall -std=c++17

SRC=BlobTests.cpp Blob.cpp
OBJ=BlobTests.o Blob.o

TARGET=test

all: $(TARGET)

# With all the object files in place, compile the final target
$(TARGET): $(OBJ)
	$(CC) $(FLAGS) $(OBJ) -o $@

# Compile all source files with $(HDR) directory included in search path
%.o: %.cpp
	$(CC) $(FLAGS) -c $< -o $@

clean:
	rm test $(OBJ)
