app = mlcraft
dirEnv = .venv
dirBin = $(dirEnv)/bin

.PHONY: help default build format test pytest typecheck formatcheck clean
.DEFAULT_GOAL = default

ifneq ($V,2)
# comment next line and Makefile will print what it's doing
.SILENT:
endif

ifndef CONFIG_PATH
CONFIG_PATH = $(abspath ../config.json)
endif
# check if file exists
ifeq ("$(wildcard $(CONFIG_PATH))","")
$(error "Not a valid config: check your CONFIG_PATH variable")
endif
ifeq ("$(shell which node)","")
$(error 'node' not found, try installing it somehow, I don't know:0)
endif

getField = $(shell node -p "require('$(CONFIG_PATH)').py_server.$1")
flags = -h $(call getField,HOST) -p $(call getField,PORT)

python = $(shell which 'python3.10' || which 'python3' || which 'python')
ifeq ("$(python)","")
$(error "Python interpreter not found, is it in your PATH?")
endif

help:
	@echo "\
	Usage: make (<target>)? (<option>=<value>)* \n\
	Example (runs tests) make test V=2 \n\
	\n\
	Available targets: \n\
	default: runs by default, starts the server \n\
	build: install + format \n\
	format: formats the code with 'black' formatter \n\
	install: updates the mlcraft package, runs automatically \
	after updating pyproject.toml \n\
	test: run all tests and typechecks \n\
	clean: cleans current instances and install flag (does not remove .venv) \n\
	\n\
	Available options: \n\
	*actually any variable used in the Makefile can be overriden by the option* \n\
	flags: you can override flags given to flask application with this \n\
	V: if V=1, will print a little info, if V=2 will print all commands \n\
	python: specify a path to a python interpreter (version must be >=3.10)
	CONFIG_PATH: path to the config file, default = ../config.json \n\
	"

default: export CONFIG_PATH := $(CONFIG_PATH)
default: install
	@echo "Starting python server"
	@echo "---------------------------------------------------------------------------"
	$(dirBin)/flask --app $(app) run --debug $(flags)

build: install

format: install
	$(dirBin)/black .

test: install formatcheck typecheck pytest
formatcheck: install
	$(dirBin)/black --diff .
typecheck: install
	$(dirBin)/mypy .
pytest: install
	$(dirBin)/pytest

install: pyproject.toml | $(dirEnv)
	$(dirBin)/pip install -e .
# dummy file to not install package every time
	touch install 

$(dirEnv):
	$(python) -m venv $@
	$(dirBin)/python -m pip install --upgrade pip
	@test -f $(dirBin)/pip || \
	 (echo "No 'pip' found in the $(dirBin), will be unable to continue" ; exit 1)

clean:
	$(RM) install
	$(RM) -r instance
