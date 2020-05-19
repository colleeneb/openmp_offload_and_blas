EXES=dgemm_cublas_fortran

CUDAC=nvcc
CC=xlc_r
FC=xlf90_r
CXXFLAGS=-O3 -qsmp -qoffload -g -lcublas
CFLAGS=$(CXXFLAGS)
CUDAFLAGS=
FFLAGS=$(CXXFLAGS)

all: $(EXES)

dgemm_cublas_fortran: cublas.f90 dgemm_cublas_fortran.f90 
	-$(FC) -o $@  $(CFLAGS) $^ 

.PHONY: clean
clean:
	rm -rf *.o *.ptx *.cub *.lst *.mod $(EXES)
