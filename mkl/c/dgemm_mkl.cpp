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


/*
  start = high_resolution_clock::now();
  for( i = 0; i < size; i++ )
  {
    #pragma omp target map(B_gpu[ptrB[i]:ptrB[i+1]])
    {
      for( j = ptrB[i]; j < ptrB[i+1]; j++ )
      {
        B_gpu[j] = 42;
      }
    }
    #pragma omp target data map(to:A[ptrA[i]:ptrA[i+1]],X[ptrX[i]:ptrX[i+1]]) map(tofrom:B_gpu[ptrB[i]:ptrB[i+1]]) device(0)
    {
      #pragma omp target variant dispatch device(0) use_device_ptr(A, X, B_gpu)
      dgemm(&trans, &trans, &M[ i ], &N[ i ], &K[ i ], &alpha, &A[ptrA[ i ]], &M[ i ], &X[ptrX[ i ]], &K[ i ], &beta, &B_gpu[ptrB[ i ]], &M[ i ]);
    }
  }
  stop = high_resolution_clock::now();
*/
// start = high_resolution_clock::now();
// for( i = 0; i < size; i++ )
//   {
//     int a_st = ptrA[i];
//     int x_st = ptrX[i];
//     int b_st = ptrB[i];
//     int a_end = ptrA[i+1] - ptrA[i];
//     int x_end = ptrX[i+1] - ptrX[i];
//     int b_end = ptrB[i+1] - ptrB[i];
//     double *a_ptr = &A[ a_st ];
//     double *x_ptr = &X[ x_st ];
//     double *b_ptr = &B_gpu[ b_st ];
// #pragma omp target map(b_ptr[0:b_end])
//     {
//       for( j = ptrB[i]; j < ptrB[i+1]; j++ )
// 	{
// 	  b_ptr[j] = 42;
// 	}
//     }
// #pragma omp target data map(to:a_ptr[0:a_end],x_ptr[0:x_end]) map(tofrom:b_ptr[0:b_end]) device(0)
//     {
// #pragma omp target variant dispatch device(0) use_device_ptr(a_ptr, x_ptr, b_ptr)
//       dgemm(&trans, &trans, &M[ i ], &N[ i ], &K[ i ], &alpha, a_ptr, &M[ i ], x_ptr, &K[ i ], &beta, b_ptr, &M[ i ]);
//     }
//   }
// stop = high_resolution_clock::now();
