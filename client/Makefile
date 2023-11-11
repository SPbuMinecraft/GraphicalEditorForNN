.PHONY: run test format
.DEFAULT_GOAL = run

ifndef CONFIG_PATH
CONFIG_PATH = $(abspath ../config.json)
endif
# check if file exists
ifeq ("$(wildcard $(CONFIG_PATH))","")
$(error Not a valid config: check your CONFIG_PATH variable)
endif
ifeq ("$(shell which npx)","")
$(error 'npx' not found, try installing 'node', see README about 'node')
endif

port = $(shell node -p "require('$(CONFIG_PATH)').client.PORT")

help:
	@echo "\n\
	Usage: make (<option>=<value>)* \n\
	Example (hosts the client code) make V=2 \n\
	\n\
	Available targets: \n\
	run: runs by default, starts the server \n\
	help: print this \n\
	\n\
	Available options: \n\
	V: if V=1, will print a little info, if V=2 will print all commands \n\
	CONFIG_PATH: path to the config file, default = ../config.json \n\
	"

run: index.html
	@echo "Starting client server"
	@echo "-----------------------------------------------------------"
	npx serve -l $(port)


formatFlags = --log-level warn

test:
	npx prettier $(formatFlags) --check .

format:
	npx prettier -w $(formatFlags) .
