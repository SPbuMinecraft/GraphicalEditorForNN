.PHONY: all debug release test clean

PORT = $(shell node -p "require('../config.json').client.PORT")
PYTHON = $(shell which 'python3' || which 'python' || (echo "python not found"; exit 1))

all: common
debug: common
release: common
build:

common: index.html
	@echo "Starting client server"
	@echo "-----------------------------------------------------------"
	python -m http.server $(PORT)

clean:
