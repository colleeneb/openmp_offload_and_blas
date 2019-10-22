EXES=sgemm_cublas sgemm_nvblas

CUDAC=nvcc
CC=xlc_r
CXXFLAGS=-O3 -qsmp -qoffload -g
CFLAGS=$(CXXFLAGS)
CUDAFLAGS=
FFLAGS=$(CXXFLAGS)
LDFLAGS=

all: $(EXES)

sgemm_cublas: sgemm.c
	$(CC) -o $@ -DCUBLAS $(CFLAGS) $^ $(LDFLAGS) -lcublas
sgemm_nvblas: sgemm.c
	$(CC) -o $@ -DNVBLAS -I/sw/summit/essl/6.1.0-2/essl/6.1/include/ $(CFLAGS) $^ $(LDFLAGS) -L${OLCF_CUDA_ROOT}/lib64  -lnvblas -L${OLCF_ESSL_ROOT}/lib64  -lessl


.SUFFIXES:
.SUFFIXES: .c .o .f90 .cu .cpp .cuf
.c.o:
	$(CC) $(CFLAGS) -c $<
.cpp.o:
	$(CXX) $(CXXFLAGS) -c $<
.f90.o:
	$(FC) $(FFLAGS) -c $<
.cuf.o:
	$(FC) $(FFLAGS) -c $<
.cu.o:
	$(CUDAC) $(CUDAFLAGS) -c $<
.PHONY: clean
clean:
	rm -rf *.o *.ptx *.cub *.lst *.mod $(EXES)
