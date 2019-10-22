# Examples
Example C code showing how to offload blas calls from OpenMP regions,
using cuBLAS and NVBLAS.
Makefile is hardcoded to run on Summit.

# Building and Running on Summit

 To run, the simplest is to submit an interactive job:

 1. Submit an interactive job:
    $ bsub -W 1:00 -nnodes 1 -P Project -Is $SHELL

 2. Set the environment:
    $ source environment_setup.sh

    This loads the cuda module, which is needed for tests
    with nvprof

 3. Compile the files
    $ make Makefile

 4. Use jsun to run each simple example:
    $ jsrun -n 1 -a 1 -c 1 -g 1 ./sgemm_nvblas

       -n is the nubmer of resource sets,
       -a is the number of MPI ranks,
       -c is the number of physical cores,
       -g is the number of GPUs

    $ jsrun -n 1 -a 1 -c 1 -g 1 nvprof --print-gpu-trace ./sgemm_cublas

     etc.
