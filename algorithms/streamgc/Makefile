# StreamGC build system
.PHONY: all streamgc test clean

all: streamgc

streamgc:
	$(MAKE) -C algorithms/streamgc all

test: streamgc
	$(MAKE) -C algorithms/streamgc unit_tests

clean:
	$(MAKE) -C algorithms/streamgc clean
