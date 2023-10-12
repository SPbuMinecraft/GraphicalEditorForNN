.PHONY: all debug release clean

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

test:
	@for dir in $(DIRS) ; do \
	( cd $$dir ; make test) ; \
	done

clean:
	@for dir in $(DIRS) ; do \
	( cd $$dir ; make clean) ; \
	done
