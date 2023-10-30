dirApi = api
dirCore = core
dirTests = tests
dirBuild = Build
dirBin = bin

# Preprocessor flags
CPPFLAGS = -I $(dirCore) -isystem $(dirApi)

# C++ compiler flags
maxErrors = 4 # number of errors compiler will print to the console
CXXFLAGS = -std=c++17 -pthread
CXXFLAGS += -Wall -Wextra -pedantic 
CXXFLAGS += -fmax-errors=$(maxErrors) # for g++
# CXXFLAGS += -ferror-limit=$(maxErrors) # uncomment for clang
CXXFLAGS += -Wno-sign-compare
CXXFLAGS += -Wno-unused-parameter
CXXFLAGS += -Wno-unused-command-line-argument
ifeq ($O,3)
CXXFLAGS += -O3
else ifeq ($O,2)
CXXFLAGS += -O2
else 
CXXFLAGS += -g
endif

# Archiver flags
ifneq ($V,2)
ifneq ($V,1)
ARFLAGS = -rc
endif
endif

ifneq ($V,2)
# comment next line and Makefile will print what it's doing
.SILENT:
endif
.DEFAULT_GOAL = default
.PHONY: default serve test test.% clean

include $(wildcard $(dirBuild)/*.d)

vpath %.cpp $(dirCore):$(dirTests):$(dirApi)

getObjects = $(patsubst $2/%.cpp, $(dirBuild)/%.o, $1)

core = $(dirBin)/core
server = $(dirBin)/server
coreMain = $(dirCore)/main.cpp
coreArchive = $(dirBuild)/libcore.a

help:
	@echo "Usage: make (<target>)? (<option>=<value>)* "
	@echo
	@echo "Example (builds default target): make O=3 V=2"
	@echo
	@echo "Available targets:"
	@echo "default: builds and runs core/main.cpp file"
	@echo "serve: builds and runs api/server.cpp file"
	@echo "test: builds and runs all tests in the tests directory"
	@echo "test.<object>: runs specific test, for example 'make test.Blob'"
	@echo "build: just build everything without running it"
	@echo "help: prints this help"
	@echo
	@echo "Available options: (actually, all variables in Makefile can be overriden by options)"
	@echo "CONFIG_PATH: path to the config file, default = ../config.json"
	@echo "port, host: if you do make serve, you can override port with this"
	@echo "V: if V=1, will print a little info, if V=2 will print all commands"
	@echo "O: if O=2, will use O2 optimizer for c++, if O=3 will use O3 optimizer"
	@echo "     can be useful to catch subtle warnings, so \033[0;31mUSE before commit"
	@echo


### BUILD ###

build: $(core) $(server) $(testTargets)


### CORE ###

coreFiles = $(wildcard $(dirCore)/*.cpp)
coreFiles := $(filter-out $(coreMain), $(coreFiles))
coreObjects = $(call getObjects,$(coreFiles),$(dirCore))

default: $(core); $<

$(core): $(coreMain) $(coreArchive) | $(dirBin)
	$(LINK.cpp) $^ -o $@

$(coreArchive): $(coreArchive)($(coreObjects)) | $(dirBuild)

.PRECIOUS: $(dirBuild)/%.o
$(dirBuild)/%.o: %.cpp | $(dirBuild)
	$(COMPILE.cpp) -MMD $< -o $@


### SERVER ###	

ifndef CONFIG_PATH
CONFIG_PATH = $(abspath ../config.json)
endif
# check if file exists
ifeq ("$(wildcard $(CONFIG_PATH))","")
$(error $(CONFIG_PATH): Not a valid config: check your CONFIG_PATH variable)
endif
ifeq ($(shell which node),)
$(error 'node' not found, try installing it somehow, I don't know:0)
endif

getField = $(shell node -p "require('$(CONFIG_PATH)').cpp_server.$1")
port = $(call getField,PORT)
host = $(call getField,HOST)
boost = $(call getField,BOOST_ROOT)

# check boost
ifeq ("$(wildcard $(boost))","")
$(warning $(boost): Not a valid boost path: check your BOOST_ROOT in the config)
endif
ifeq ("$(wildcard $(boost)/lib)","")
$(warning $(boost): No `lib` folder found: check BOOST_ROOT in the config)
endif
ifeq ("$(wildcard $(boost)/include)","")
$(warning $(boost): No `include` folder found: check BOOST_ROOT in the config)
endif
CPPFLAGS += -isystem $(boost)/include

serverFiles = $(wildcard $(dirApi)/*.cpp)
serverObjects = $(call getObjects,$(serverFiles),$(dirApi))

serve: $(server); $< $(host) $(port)

$(server): $(serverObjects) $(coreArchive) | $(dirBin)
	$(LINK.cpp) -L $(boost)/lib $^ -o $@


### TESTS ###

testFiles = $(wildcard $(dirTests)/*Tests.cpp)
testNames = $(testFiles:$(dirTests)/%Tests.cpp=%)
testTargets = $(testNames:%=$(dirBin)/test%)

test: $(foreach name, $(testNames), test.$(name))
	@echo "All tests completed"

.PRECIOUS: $(dirBin)/test%
test.%: $(dirBin)/test%
	$<
	@echo "$* test completed"

$(dirBin)/test%: $(dirBuild)/%Tests.o $(coreArchive) | $(dirBin)
	$(LINK.cpp) $^ -o $@


### COMMON ###

$(dirBuild) $(dirBin): ; mkdir $@

clean: ; $(RM) -r $(dirBin) $(dirBuild)

# api: 
# 	crow.h
# 	server.cpp
#   session.h
#   session.cpp
# core: 
# 	Tensor.h
# 	Tensor.cpp
# 	main.cpp
# tests:
#	doctest.h
#   BlobTests.cpp
# 	TensorTests.cpp
# build:
# 	Tensor.o
# 	server.o
# 	TensorTests.o
#   Tensor.d
#   TensorTests.d
# bin:
# 	server
# 	core
# 	testTensor
#   testBlob	
