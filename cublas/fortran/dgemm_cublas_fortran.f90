
program main
  use cublasf
  implicit none
  double precision, allocatable, target, dimension(:) :: aa
  double precision, allocatable, target, dimension(:) :: bb
  double precision, allocatable, target, dimension(:) :: cc_gpu
  double precision, allocatable, target, dimension(:) :: cc_host

  integer :: status, SIZE, i, ic, ib, ia, error
  integer(c_int)   :: cublas_error
  type(c_ptr)      :: cublas_handle

  double precision, target :: alpha, beta
  error = 0
  alpha = 1d0
  beta = 1d0
  SIZE = 1024

! create cublas handle
  cublas_error = cublascreate_v2(cublas_handle)

  allocate(aa(SIZE*SIZE))
  allocate(bb(SIZE*SIZE))
  allocate(cc_gpu(SIZE*SIZE))
  allocate(cc_host(SIZE*SIZE))

  call srand(123456)

  do i=1,SIZE*SIZE
     aa(i) = rand()*10
     bb(i) = rand()*10
     cc_host(i) = 0d0
     cc_gpu(i) = 0d0
  enddo

! get value on host
  do ic=1,SIZE
     do ib=1,SIZE
        cc_host(ib+(ic-1)*SIZE) = 0d0
        do ia=1,SIZE
           cc_host(ib+(ic-1)*SIZE) = cc_host(ib+(ic-1)*SIZE)+aa(ib+(ia-1)*SIZE)*bb(ia+(ic-1)*SIZE)
        enddo
     enddo
  enddo

! get value from gpu

!$omp target enter data map(to:aa,bb,cc_gpu)

!$omp target data use_device_ptr(aa,bb,cc_gpu)
  cublas_error = cublasDgemm_v2(cublas_handle,CUBLAS_OP_N, CUBLAS_OP_N, SIZE, SIZE,SIZE, alpha, c_loc(aa), &
       SIZE, c_loc(bb), SIZE, beta, c_loc(cc_gpu), SIZE);
!$omp end target data
   if(cublas_error .ne. CUBLAS_STATUS_SUCCESS ) then
      print *, "failed", cublas_error, cc_gpu(1)
      call exit(1)
   endif

   ! wait for call to finish
   cublas_error = cudaDeviceSynchronize_v2()
   cublas_error = cublasDestroy_v2(cublas_handle)

!$omp target exit data map(from:cc_gpu)

! error checking
  do i=1,SIZE*SIZE
     if( abs( cc_gpu(i) - cc_host(i) ) > 0.00001 ) then
        error=error+1
     endif
  enddo

  if( error > 0 ) then
     print *, "Failed" 
     call exit(1)
  endif

  deallocate( aa, stat=status )
  deallocate( bb, stat=status )
  deallocate( cc_host, stat=status )
  deallocate( cc_gpu, stat=status )

end program main
