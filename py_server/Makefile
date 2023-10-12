.PHONY: all debug release test clean

HOST = $(shell node -p "require('../config.json').py_server.HOST")
PORT = $(shell node -p "require('../config.json').py_server.PORT")
PYTHON = $(shell which 'python3' || which 'python' || (echo "python not found"; exit 1))
FLAGS = -h $(HOST) -p $(PORT)
APP = mlcraft
ENVDIR = .venv
BIN = $(ENVDIR)/bin

all: debug

debug: install
	@echo "Starting python server"
	@echo "---------------------------------------------------------------------------"
	flask --app $(APP) run --debug $(FLAGS)

release: install
	@$(BIN)/pip install waitress
	@$(BIN)/waitress-serve --host=$(HOST) --port=$(PORT) --call '$(APP):make_app' 

test: install
	@$(BIN)/pytest

install: pyproject.toml $(ENVDIR)
	$(BIN)/pip install -e .
# dummy file to not install package every time
	touch install 

env:
	$(PYTHON) -m venv $(ENVDIR)

clean:
	@rm install
	@rm -fr instance
