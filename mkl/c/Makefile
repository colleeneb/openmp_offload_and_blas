EXES=dgemm_mkl cblas_dgemm_mkl

CC=icx
FC=ifort
CXXFLAGS=-qnextgen -mllvm -vpo-paropt-use-interop -fiopenmp -fopenmp-targets=spir64 -g -mkl
CFLAGS=$(CXXFLAGS)
CUDAFLAGS=
FFLAGS=$(CXXFLAGS)

all: $(EXES)

dgemm_mkl: dgemm_mkl.cpp
	-$(CC) -o $@  $(CFLAGS) $^  
cblas_dgemm_mkl: cblas_dgemm_mkl.cpp
	-$(CC) -o $@  $(CFLAGS) $^  

.PHONY: clean
clean:
	rm -rf *.o *.ptx *.cub *.lst *.mod $(EXES)
