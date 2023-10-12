.PHONY: debug release test clean
# Compiler
CC = g++
# Compiler flags
FLAGS = -g -Wall -std=c++17
RELEASE_FLAGS = -O3

HDR1 = api
HDR2 = core
SRC = $(wildcard api/*.cpp) $(wildcard core/*.cpp)
OBJ = $(SRC:.cpp=.o)
RELEASE_OBJ = $(SRC:.cpp=_release.o)

TEST_SRC = $(wildcard tests/*.cpp) $(wildcard core/*.cpp)
TEST_OBJ = $(TEST_SRC:.cpp=.o)

DEBUG_TARGET = cpp_server
RELEASE_TARGET = cpp_server_release # the same thing
TEST_TARGET = cpp_tests

all: debug # Default is debug target

debug: $(DEBUG_TARGET)
	@echo "Starting cpp server in debug"
	@echo "------------------------------------------------------------------"
	./$(DEBUG_TARGET)

$(DEBUG_TARGET): $(OBJ)
	$(CC) $(FLAGS) $^ -o $(DEBUG_TARGET)

release: $(RELEASE_TARGET)
	@echo "Starting cpp server in release"
	@echo "------------------------------------------------------------------"
	./$(RELEASE_TARGET)

$(RELEASE_TARGET): $(RELEASE_OBJ)
	$(CC) $(FLAGS) $(RELEASE_FLAGS) $^ -o $(RELEASE_TARGET)

%_release.o: %.cpp
	$(CC) $(FLAGS) $(RELEASE_FLAGS) -I$(HDR1) -I$(HDR2) -c $< -o $@

%.o: %.cpp
	$(CC) $(FLAGS) -I$(HDR1) -I$(HDR2) -c $< -o $@

test: $(TEST_TARGET)
	@echo "Testing cpp server"
	@echo "------------------------------------------------------------------"
	./$(TEST_TARGET)
	@echo "------------------------------------------------------------------"

$(TEST_TARGET): $(TEST_OBJ)
	$(CC) $(FLAGS) $^ -o $@

clean:
	@rm -f $(OBJ)
	@rm -f $(RELEASE_OBJ)
	@rm -f $(TEST_OBJ)
	@rm -f $(RELEASE_TARGET) $(DEBUG_TARGET) $(TEST_TARGET)
	@rm -f test
