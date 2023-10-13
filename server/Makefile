.PHONY: debug release test clean
# Compiler
CC = g++
# Compiler flags
RELEASE_FLAGS = -O3

FLAGS = -g -Wall -std=c++17 -isystemapi -isystemcore -isystem$(BOOST_ROOT)/include -L$(BOOST_ROOT)/lib
BIN = bin

SRC = $(wildcard api/*.cpp) $(wildcard core/*.cpp)
OBJ = $(SRC:.cpp=.o)
RELEASE_OBJ = $(SRC:.cpp=_release.o)

TEST_SRC = $(wildcard tests/*.cpp) $(wildcard core/*.cpp)
TEST_OBJ = $(TEST_SRC:.cpp=.o)

DEBUG_TARGET = $(BIN)/cpp_server
RELEASE_TARGET = $(BIN)/cpp_server_release
TEST_TARGET = $(BIN)/cpp_tests

all: debug # Default is debug target

debug: $(DEBUG_TARGET)
	@echo "Starting cpp server in debug"
	@echo "------------------------------------------------------------------"
	./$(DEBUG_TARGET)

$(DEBUG_TARGET): $(OBJ) $(BIN)
	$(CC) $(FLAGS) $(OBJ) -o $(DEBUG_TARGET)

release: $(RELEASE_TARGET)
	@echo "Starting cpp server in release"
	@echo "------------------------------------------------------------------"
	./$(RELEASE_TARGET)

build: $(DEBUG_TARGET)

$(RELEASE_TARGET): $(RELEASE_OBJ) $(BIN)
	$(CC) $(FLAGS) $(RELEASE_FLAGS) $(RELEASE_OBJ) -o $(RELEASE_TARGET)

%_release.o: %.cpp
	$(CC) $(FLAGS) $(RELEASE_FLAGS) -c $< -o $@

%.o: %.cpp
	$(CC) $(FLAGS) -c $< -o $@

test: $(TEST_TARGET)
	@echo "Testing cpp server"
	@echo "------------------------------------------------------------------"
	./$(TEST_TARGET)
	@echo "------------------------------------------------------------------"

$(TEST_TARGET): $(TEST_OBJ) $(BIN)
	$(CC) $(FLAGS) $(TEST_OBJ) -o $@

$(BIN):
	mkdir $(BIN)

clean:
	@rm -f $(OBJ)
	@rm -f $(RELEASE_OBJ)
	@rm -f $(TEST_OBJ)
	@rm -f $(RELEASE_TARGET) $(DEBUG_TARGET) $(TEST_TARGET)
	@rm -f test
