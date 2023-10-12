.PHONY: all debug release build clean

DIRS = client py_server server
MAKEFLAGS += -j6

all: debug

debug:
	@for dir in $(DIRS) ; do \
	( cd $$dir ; make debug &) ; \
	done

release:
	@for dir in $(DIRS) ; do \
	( cd $$dir ; make release &) ; \
	done

build:
	@for dir in $(DIRS) ; do \
	( cd $$dir ; make build) ; \
	done

test:
	@for dir in $(DIRS) ; do \
	( cd $$dir ; make test) ; \
	done

clean:
	@for dir in $(DIRS) ; do \
	( cd $$dir ; make clean) ; \
	done
