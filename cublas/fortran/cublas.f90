!  Copyright (C) 2020, Argonne National Laboratory. All Rights Reserved.
!  Licensed under the NCSA open source license

module cublasf
  !-------------------------------------------------------------------------------------------------
  ! Interface to cuBLAS routines
  !-------------------------------------------------------------------------------------------------
  use, intrinsic :: iso_c_binding

  enum, bind(c) !:: cublasStatus_t
    enumerator :: CUBLAS_STATUS_SUCCESS = 0
    enumerator :: CUBLAS_STATUS_NOT_INITIALIZED = 1
    enumerator :: CUBLAS_STATUS_ALLOC_FAILED = 3
    enumerator :: CUBLAS_STATUS_INVALID_VALUE = 7
    enumerator :: CUBLAS_STATUS_ARCH_MISMATCH = 8
    enumerator :: CUBLAS_STATUS_MAPPING_ERROR = 11
    enumerator :: CUBLAS_STATUS_EXECUTION_FAILED = 13
    enumerator :: CUBLAS_STATUS_INTERNAL_ERROR = 14
  end enum !cublasStatus_t

  enum, bind(c) !:: cublasFillMode_t
    enumerator :: CUBLAS_FILL_MODE_LOWER = 0
    enumerator :: CUBLAS_FILL_MODE_UPPER = 1
  end enum !cublasFillMode_t

  enum, bind(c) !:: cublasDiag type_t
    enumerator :: CUBLAS_DIAG_NON_UNIT = 0
    enumerator :: CUBLAS_DIAG_UNIT = 1
  end enum !cublasDiag    type_t

  enum, bind(c) !:: cublasSideMode_t
    enumerator :: CUBLAS_SIDE_LEFT = 0
    enumerator :: CUBLAS_SIDE_RIGHT = 1
  end enum !cublasSideMode_t

  enum, bind(c) !:: cublasOperation_t
    enumerator :: CUBLAS_OP_N = 0
    enumerator :: CUBLAS_OP_T = 1
    enumerator :: CUBLAS_OP_C = 2
  end enum !cublasOperation_t

  interface

    integer(c_int) function &
         cublasInit() &
         bind(c, name="cublasInit")
      use, intrinsic :: iso_c_binding
    end function cublasInit

    integer(c_int) function &
         cublasShutdown() &
         bind(c, name="cublasShutdown")
      use, intrinsic :: iso_c_binding
    end function cublasShutdown

    integer(c_int) function &
         cublasCreate_v2(handle) &
         bind(c, name="cublasCreate_v2")
      use, intrinsic :: iso_c_binding
      type(c_ptr) :: handle
    end function cublasCreate_v2

    integer(c_int) function &
         cublasDestroy_v2(handle) &
         bind(c, name="cublasDestroy_v2")
      use, intrinsic :: iso_c_binding
      type(c_ptr), value :: handle
    end function cublasDestroy_v2

    integer(c_int) function &
         cublasGetStream_v2(handle, stream) &
         bind(c, name="cublasGetStream_v2")
      use, intrinsic :: iso_c_binding
      type(c_ptr), value :: handle
      type(c_ptr) :: stream
    end function cublasGetStream_v2

    integer(c_int) function &
         cublasSetStream_v2(handle, stream) &
         bind(c, name="cublasSetStream_v2")
      use, intrinsic :: iso_c_binding
      type(c_ptr), value :: handle
      type(c_ptr), value :: stream
    end function cublasSetStream_v2

    integer(c_int) function &
         cudaDeviceSynchronize_v2() &
         bind(c, name="cudaDeviceSynchronize")
      use, intrinsic :: iso_c_binding
    end function cudaDeviceSynchronize_v2

    integer(c_int) function &
         cublasDgemmBatched(handle, transa, transb, m, n, k, alpha, dA,&
         ldda, dB, lddb, beta, dC, lddc, nbatch) &
         bind(c, name="cublasDgemmBatched")
      use, intrinsic :: iso_c_binding
      type(c_ptr), value :: handle
      integer(c_int), value :: transa
      integer(c_int), value :: transb
      integer(c_int), value :: m
      integer(c_int), value :: n
      integer(c_int), value :: k
      real(c_double) :: alpha
      type(c_ptr), value :: dA
      integer(c_int), value :: ldda
      type(c_ptr), value :: dB
      integer(c_int), value :: lddb
      real(c_double) :: beta
      type(c_ptr), value :: dC
      integer(c_int), value :: lddc
      integer(c_int), value :: nbatch
    end function cublasDgemmBatched

    integer(c_int) function &
         cublasDgemmStridedBatched(handle, transa, transb, m, n, k, &
         alpha, &
         dA, ldda, strideA, dB, lddb, strideB, beta, dC, lddc, &
         strideC,nbatch) &
         bind(c, name="cublasDgemmStridedBatched")
      use, intrinsic :: iso_c_binding
      type(c_ptr), value :: handle
      integer(c_int), value :: transa
      integer(c_int), value :: transb
      integer(c_int), value :: m
      integer(c_int), value :: n
      integer(c_int), value :: k
      real(c_double) :: alpha
      type(c_ptr), value :: dA
      integer(c_int), value :: ldda
      integer(c_int), value :: strideA
      type(c_ptr), value :: dB
      integer(c_int), value :: lddb
      integer(c_int), value :: strideB
      real(c_double) :: beta
      type(c_ptr), value :: dC
      integer(c_int), value :: strideC
      integer(c_int), value :: lddc
      integer(c_int), value :: nbatch
    end function cublasDgemmStridedBatched

    ! integer(c_int) function &
    !      cublasDgemm(handle, transa, transb, m, n, k, alpha, dA, ldda, dB,&
    !      lddb, beta, dC, lddc) &
    !      bind(c, name="cublasDgemm")
    !   use, intrinsic :: iso_c_binding
    !   type(c_ptr), value :: handle
    !   character(c_char), value :: transa
    !   character(c_char), value :: transb
    !   integer(c_int), value :: m
    !   integer(c_int), value :: n
    !   integer(c_int), value :: k
    !   real(c_double) :: alpha
    !   type(c_ptr), value :: dA
    !   integer(c_int), value :: ldda
    !   type(c_ptr), value :: dB
    !   integer(c_int), value :: lddb
    !   real(c_double) :: beta
    !   type(c_ptr), value :: dC
    !   integer(c_int), value :: lddc
    ! end function cublasDgemm

    integer(c_int) function &
         cublasDgemm_v2(handle, transa, transb, m, n, k, alpha, dA,&
         ldda, dB, lddb, beta, dC, lddc) &
         bind(c, name="cublasDgemm_v2")
      use, intrinsic :: iso_c_binding
      type(c_ptr), value :: handle
      integer(c_int), value :: transa
      integer(c_int), value :: transb
      integer(c_int), value :: m
      integer(c_int), value :: n
      integer(c_int), value :: k
      real(c_double) :: alpha
!      type(c_ptr) :: alpha
      !real(c_double),dimension(*) :: dA, dB, dC
      type(c_ptr), value :: dA
      integer(c_int), value :: ldda
      type(c_ptr), value :: dB
      integer(c_int), value :: lddb
      real(c_double) :: beta
!      type(c_ptr) :: beta
      type(c_ptr), value :: dC
      integer(c_int), value :: lddc
    end function cublasDgemm_v2

    integer(c_int) function &
         cublasSgemm_v2(handle, transa, transb, m, n, k, alpha, dA,&
         ldda, dB, lddb, beta, dC, lddc) &
         bind(c, name="cublasSgemm_v2")
      use, intrinsic :: iso_c_binding
      type(c_ptr), value :: handle
      integer(c_int), value :: transa
      integer(c_int), value :: transb
      integer(c_int), value :: m
      integer(c_int), value :: n
      integer(c_int), value :: k
      real(c_float) :: alpha
!      type(c_ptr) :: alpha
      !real(c_double),dimension(*) :: dA, dB, dC
      type(c_ptr), value :: dA
      integer(c_int), value :: ldda
      type(c_ptr), value :: dB
      integer(c_int), value :: lddb
      real(c_float) :: beta
!      type(c_ptr) :: beta
      type(c_ptr), value :: dC
      integer(c_int), value :: lddc
    end function cublasSgemm_v2

    integer(c_int) function &
         cublasSgemm_v3(transa, transb, m, n, k, alpha, dA,&
         ldda, dB, lddb, beta, dC, lddc) &
         bind(c, name="cublasSgemm_v3")
      use, intrinsic :: iso_c_binding
      integer(c_int), value :: transa
      integer(c_int), value :: transb
      integer(c_int), value :: m
      integer(c_int), value :: n
      integer(c_int), value :: k
      real(c_float) :: alpha
      !real(c_float),dimension(*) :: dA, dB, dC
      type(c_ptr), value :: dA
      integer(c_int), value :: ldda
      type(c_ptr), value :: dB
      integer(c_int), value :: lddb
      real(c_float),value :: beta
      type(c_ptr), value :: dC
      integer(c_int), value :: lddc
    end function cublasSgemm_v3


    integer(c_int) function &
         cublasxtcreate(handle) &
         bind(c, name="cublasXtCreate")
      use, intrinsic :: iso_c_binding
      type(c_ptr) :: handle
    end function cublasxtcreate

    integer(c_int) function &
         cublasxtdestroy(handle) &
         bind(c, name="cublasXtDestroy")
      use, intrinsic :: iso_c_binding
      type(c_ptr), value :: handle
    end function cublasxtdestroy

    integer(c_int) function &
         cublasXtDeviceSelect(handle, nbDevices, deviceId) &
         bind(c, name="cublasXtDeviceSelect")
      use, intrinsic :: iso_c_binding
      type(c_ptr), value :: handle
      integer(c_int), value :: nbDevices
      integer(c_int),dimension(*) :: deviceId
    end function cublasXtDeviceSelect

    integer(c_int) function &
         cublasXtSetBlockDim(handle, blockDim) &
         bind(c, name="cublasXtSetBlockDim")
      use, intrinsic :: iso_c_binding
      type(c_ptr), value :: handle
      integer(c_int), value :: blockDim
    end function cublasXtSetBlockDim

    integer(c_int) function &
         cublasXtDgemm(handle, transa, transb, m, n, k, alpha, dA,&
         ldda, dB, lddb, beta, dC, lddc) &
         bind(c, name="cublasXtDgemm")
      use, intrinsic :: iso_c_binding
      type(c_ptr), value :: handle
      integer(c_int), value :: transa
      integer(c_int), value :: transb
      integer(c_int), value :: m
      integer(c_int), value :: n
      integer(c_int), value :: k
      real(c_double) :: alpha
      real(c_double),dimension(*) :: dA, dB, dC
      !type(c_ptr), value :: dA
      integer(c_int), value :: ldda
      !type(c_ptr), value :: dB
      integer(c_int), value :: lddb
      real(c_double) :: beta
      !type(c_ptr), value :: dC
      integer(c_int), value :: lddc
    end function cublasXtDgemm

  end interface

end module cublasf
