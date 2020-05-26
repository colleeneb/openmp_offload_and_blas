#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#define SIZE 1024

int dnum = 0;

int main( int argc, char* argv[] )
{
  double *aa;
  double *bb;
  double *cc_gpu;
  double *cc_host;
  int *offsets;

  int teams = 0;
  int threads = 0;
  int error = 0;
  int size = SIZE;
  int num_dgemms = 2;

  double alpha = 1.0;
  double beta = 1.0;
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
    {
      fprintf(stdout, "CUBLAS initialization failed!\n");
      exit(EXIT_FAILURE);
    }

  if( (cc_gpu = (double *)malloc( sizeof(double)*SIZE*SIZE*num_dgemms)) == NULL ) {
    printf("problem\n");
  }
  if( (cc_host = (double *)malloc( sizeof(double)*SIZE*SIZE*num_dgemms)) == NULL ) {
    printf("problem\n");
  }
  if( (aa = (double *)malloc( sizeof(double)*SIZE*SIZE*num_dgemms)) == NULL ) {
    printf("problem\n");
  }
  if( (bb = (double *)malloc( sizeof(double)*SIZE*SIZE*num_dgemms)) == NULL ) {
    printf("problem\n");
  }
  if( (offsets = (int *)malloc( sizeof(int)*num_dgemms)) == NULL ) {
    printf("problem\n");
  }


  for(int i=0;i<SIZE*SIZE*num_dgemms;i++)
    {
      aa[i] = rand()%10;
      bb[i] = rand()%10;
      cc_host[i] = 0;
      cc_gpu[i] = 0;
    }

  // array with slices
  offsets[0] = 0;
  for( int i=1; i<num_dgemms ; i++)
    {
      offsets[i] = SIZE*SIZE*i;
    }

   // get value on host                                                                                              
for( int n=0; n< num_dgemms;n++)
    {                                                                                                            
  for(int i=0; i<SIZE; ++i) {
    for(int j=0; j<SIZE; ++j) {
      for(int k=0; k<SIZE; ++k) {
	cc_host[offsets[n]+i+j*SIZE] += aa[offsets[n]+i+k*SIZE] * bb[offsets[n]+k+j*SIZE];
      }
    }
  }
    }


  // try batching it yourself
  for(int i=0; i< num_dgemms; i++)
    {
      
      double *temp_aa = &aa[offsets[i]];
      double *temp_bb = &bb[offsets[i]];
      double *temp_cc = &cc_gpu[offsets[i]];

#pragma omp target enter data \
  map(to:temp_aa[0:SIZE*SIZE],temp_bb[0:SIZE*SIZE],temp_cc[0:SIZE*SIZE])

#pragma omp target data use_device_ptr( temp_aa, temp_bb, temp_cc) 
      {
	int cublas_error = cublasDgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N,
				       size, size, size, &alpha, temp_aa,
				       size, temp_bb,
				       size, &beta,
				       temp_cc, size);
	if( cublas_error != CUBLAS_STATUS_SUCCESS )
	  {
	    printf( "failed %d.\n", cublas_error );
	    exit(1);
	  }
      }
      // wait for call to finish
      cudaDeviceSynchronize(); // Avoid putting synchronization if not needed.

#pragma omp target exit data map(from:temp_cc[0:SIZE*SIZE])
    }

  cublasDestroy(handle);

  // error checking
      for(int i=0;i<SIZE*SIZE*num_dgemms;i++)
	{
	  if( fabs(cc_gpu[i] - cc_host[i]) > 0.0000001 )
	    {
	      if(i == 0) printf( "%f %f\n", cc_gpu[i], cc_host[i] );
	      error++;
	    }
	}

  if( error > 0 ) printf( "Failed!\n" );

  assert(error == 0);
  free(aa);
  free(bb);
  free(cc_gpu);
  free(cc_host);
  free(offsets);
  return 0;
}
