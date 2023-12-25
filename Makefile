.PHONY: help docker build.% test.% clean
.DEFAULT_GOAL = help

targets = py_server server

export CONFIG_PATH = $(abspath config.json)

ifeq ($(shell which node),)
$(error 'node' not found, see README about node)
endif

clientPort = $(shell node -p "require('$(CONFIG_PATH)').client.PORT")
pythonPort = $(shell node -p "require('$(CONFIG_PATH)').py_server.PORT")
# pythonPort = $(call getField,PORT)

MAKEFLAGS += -k
ifneq ($V,2)
# comment next line and Makefile will print what it's doing
.SILENT:
MAKEFLAGS += --no-print-directory
endif

define HELP
Usage: make (build|test).(py_server|server) (<option>=<value>)*
       make run.(client|py_server|server) (<option>=<value)*
       make clean (<option>=<value>)*
Example (runs tests in py_server): make test.py_server V=2

Targets:
--------
docker: launches everything in 3 docker containers
test.docker: start docker containers, run tests in them and exit
run.client: starts the client
run.py_server: starts the py_server
run.server: starts the c++ server
build.py_server: builds py_server
build.server: builds server
test.py_server: invokes tests in py_server
test.server: invokes test in server
help: print this message
clean: cleans in server and py_server

Available options: 
*They are all will be passed to the lower Makefiles* 
------
CONFIG_PATH: path to the config file, default = 'config.json'
V: if V=1, will print a little info, if V=2 will print all commands 
O: if O=2, will use O2 optimizer for c++, if O=3 will use O3 optimizer
boost: path to the boost root directory, default is taken from config
python: specify a path to a python interpreter (version must be >=3.10)
flags: you can override flags given to flask application with this 

endef

help: export HELP := $(HELP)
help:
	@echo "$$HELP"

docker: export WEBPORT=$(clientPort)
docker: export PYPORT=$(pythonPort)
docker: compose.yml
	docker compose up

test.docker.py_server:
	$(MAKE) docker -C py_server -e

test.docker.server:
	$(MAKE) docker -C server -e

$(addprefix run.,py_server client): run.%:
	$(MAKE) -C $* -e

run.server:
	$(MAKE) serve -C server -e

$(addprefix build.,$(targets)): build.%:
	$(MAKE) -C $* -e build 

$(addprefix test.,$(targets) client): test.%:
	$(MAKE) -C $* -e test

clean:
	@for dir in $(targets) ; do \
		$(MAKE) -C $$dir clean ; \
	done
