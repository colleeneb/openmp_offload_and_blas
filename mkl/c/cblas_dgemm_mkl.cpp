#include "stdio.h"
#include "mkl.h"
#include "assert.h"
#include "math.h"
#include "mkl_omp_offload.h"

#define SIZE 1024

int main( int argc, char* argv[] )
{
  double *aa;
  double *bb;
  double *cc_gpu;
  double *cc_host;

  int teams = 0;
  int threads = 0;
  int error = 0;

  double alpha = 1.0;
  double beta = 1.0;

  if( (cc_gpu = (double *)malloc( sizeof(double)*SIZE*SIZE)) == NULL ) {
    printf("problem\n");
  }
  if( (cc_host = (double *)malloc( sizeof(double)*SIZE*SIZE)) == NULL ) {
    printf("problem\n");
  }
  if( (aa = (double *)malloc( sizeof(double)*SIZE*SIZE)) == NULL ) {
    printf("problem\n");
  }
  if( (bb = (double *)malloc( sizeof(double)*SIZE*SIZE)) == NULL ) {
    printf("problem\n");
  }


  for(int i=0;i<SIZE*SIZE;i++)
    {
      aa[i] = rand()%10;
      bb[i] = rand()%10;
      cc_host[i] = 0;
      cc_gpu[i] = 0;
    }


   // get value on host                                                                                                                                                                                                        
  for(int i=0; i<SIZE; ++i) {
    for(int j=0; j<SIZE; ++j) {
      for(int k=0; k<SIZE; ++k) {
	cc_host[i+j*SIZE] += aa[i+k*SIZE] * bb[k+j*SIZE];
      }
    }
  }

#pragma omp target enter data map(to:aa[0:SIZE*SIZE],bb[0:SIZE*SIZE],cc_gpu[0:SIZE*SIZE])

#pragma omp target variant dispatch use_device_ptr(aa, bb, cc_gpu) 
      cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,SIZE,SIZE,SIZE, alpha, aa,
		   SIZE, bb, SIZE, beta, cc_gpu, SIZE);

 #pragma omp target exit data map(from:cc_gpu[0:SIZE*SIZE])

  // error checking
  for(int i=0;i<SIZE*SIZE;i++)
    {
      if( fabs(cc_gpu[i] - cc_host[i]) > 0.0000001 )
        {
          if( i == 0) printf( "%f %f\n", cc_gpu[i], cc_host[i] );
	  error++;
        }

    }

  if( error > 0 ) printf( "Failed!\n" );

  assert(error == 0);
  free(aa);
  free(bb);
  free(cc_gpu);
  free(cc_host);

  return 0;
}
