# OpenMP Offload/Blas Examples
Example C code showing how to offload blas calls from OpenMP regions,
using cuBLAS and NVBLAS.

Two executables, `sgemm_nvblas` and `sgemm_cublas` should be produced.
`sgemm_nvblas` uses NVBLAS, and `sgemm_cublas` explicitly uses cuBLAS

A snippet of the code, showing the interface is:
```
#pragma omp target data map(to:aa[0:SIZE*SIZE],bb[0:SIZE*SIZE],alpha,beta) map(tofrom:cc_gpu[0:SIZE*SIZE]) use_device_ptr(aa,bb,cc_gpu)
 {
#if defined(CUBLAS)
   cublasSgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N,SIZE, SIZE, SIZE, &alpha, aa, SIZE, bb, SIZE, &beta, cc_gpu, SIZE);
#endif
#if defined(NVBLAS)
   sgemm("N","N",&size, &size, &size, &alpha, aa, &size, bb, &size, &beta, cc_gpu, &size);
#endif
  }
```

Makefile is hardcoded to run on Summit.

# Building and Running on Summit

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

 3. Compile the files (one using cuBLAS and NVBLAS)
    ```
    $ make Makefile
    ```
    Two executables, `sgemm_nvblas` and `sgemm_cublas` should be produced.
    `sgemm_nvblas` uses NVBLAS, and `sgemm_cublas` explicitly uses cuBLAS
 4. Use jsun to run each simple example:
    ```
    $ jsrun -n 1 -a 1 -c 1 -g 1 ./sgemm_nvblas

       -n is the nubmer of resource sets,
       -a is the number of MPI ranks,
       -c is the number of physical cores,
       -g is the number of GPUs

    $ jsrun -n 1 -a 1 -c 1 -g 1 nvprof --print-gpu-trace ./sgemm_cublas
    ```
     etc.
