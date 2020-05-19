EXES=dgemm_nvblas

CUDAC=nvcc
CC=xlc_r
FC=xlf90_r
CXXFLAGS=-O3 -qsmp -qoffload -g -lcublas
CFLAGS=$(CXXFLAGS)
CUDAFLAGS=
FFLAGS=$(CXXFLAGS)

all: $(EXES)

dgemm_nvblas: dgemm_nvblas.c
	$(CC) -o $@  $(CFLAGS) $^ $(LDFLAGS) -L${OLCF_CUDA_ROOT}/lib64  -lnvblas -L${OLCF_ESSL_ROOT}/lib64 -lessl

.PHONY: clean
clean:
	rm -rf *.o *.ptx *.cub *.lst *.mod $(EXES)
