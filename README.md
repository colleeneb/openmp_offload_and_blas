# OpenMP Offload/Blas Examples
Example C and Fortran code showing how to offload blas calls from OpenMP regions,
using cuBLAS, NVBLAS, and MKL.

There are three directories: 
 - cublas
 - nvblas
 - mkl

These contain Makefiles and examples of calling DGEMM from an OpenMP
offload region with cuBLAS, NVBLAS, and MKL. Note: The NVBLAS Makefile is hard-coded for Summit.

To run them, cd into cublas, nvblas, or mkl, then cd into a "c" or "fortran" subdirectory,
and compile with make.

 - `dgemm_nvblas` uses NVBLAS
 - `dgemm_cublas` explicitly uses cuBLAS
 - `dgemm_cublas_fortran` calls cuBLAS from Fortran using iso_c bindings. 
 - `dgemm_mkl` uses MKL with the Intel extension `target variant dispatch`

A snippet of the code, showing the interface is:

For cublas:
```
#pragma omp target data map(to:aa[0:SIZE*SIZE],bb[0:SIZE*SIZE],alpha,beta) map(tofrom:cc_gpu[0:SIZE*SIZE]) use_device_ptr(aa,bb,cc_gpu)
 {
   cublasDgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N,SIZE, SIZE, SIZE, &alpha, aa, SIZE, bb, SIZE, &beta, cc_gpu, SIZE);
  }
```

For nvblas:
```
#pragma omp target data map(to:aa[0:SIZE*SIZE],bb[0:SIZE*SIZE],alpha,beta) map(tofrom:cc_gpu[0:SIZE*SIZE]) use_device_ptr(aa,bb,cc_gpu)
 {
   dgemm("N","N",&size, &size, &size, &alpha, aa, &size, bb, &size, &beta, cc_gpu, &size);
  }
```

For mkl:
```
#pragma omp target variant dispatch use_device_ptr(aa, bb, cc_gpu) 
      cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,SIZE,SIZE,SIZE, alpha, aa,
		   SIZE, bb, SIZE, beta, cc_gpu, SIZE);
```

# Building CuBLAS and NVBLAS examples on Summit

 To run, the simplest is to submit an interactive job:

 1. Submit an interactive job:
    ```
    $ bsub -W 1:00 -nnodes 1 -P Project -Is $SHELL
    ```
 2. Set the environment:
    ```
    $ source environment_setup.sh
    ```
    This loads the cuda module, which is needed for tests
    with nvprof

 3. Move into a subdirectory (e.g. nvblas/c or cublas/fortran)
    ```
    $ cd cublas/fortran
    ``` 

 3. Compile the file
    ```
    $ make 
    ```

 4. Use jsun to run each simple example:
    ```
    $ jsrun -n 1 -a 1 -c 1 -g 1 ./dgemm_nvblas

       -n is the nubmer of resource sets,
       -a is the number of MPI ranks,
       -c is the number of physical cores,
       -g is the number of GPUs

    $ jsrun -n 1 -a 1 -c 1 -g 1 nvprof --print-gpu-trace ./dgemm_cublas
    ```
     etc.
