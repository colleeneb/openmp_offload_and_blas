EXES=dgemm_cublas

CUDAC=nvcc
CC=xlc_r
FC=xlf90_r
CXXFLAGS=-O3 -qsmp -qoffload -g -lcublas
CFLAGS=$(CXXFLAGS)
CUDAFLAGS=
FFLAGS=$(CXXFLAGS)

all: $(EXES)

dgemm_cublas: dgemm_cublas.c
	-$(CC) -o $@  $(CFLAGS) $^  

.PHONY: clean
clean:
	rm -rf *.o *.ptx *.cub *.lst *.mod $(EXES)
