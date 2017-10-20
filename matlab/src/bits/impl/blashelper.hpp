// @file blashelper.hpp
// @brief BLAS helpers
// @author Andrea Vedaldi

/*
Copyright (C) 2015-17 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__blashelper__
#define __vl__blashelper__

#include "../data.hpp"

#ifdef APPLE_BLAS
#include <Accelerate/Accelerate.h>
#else
#include <blas.h>
#endif

#ifdef ENABLE_GPU
#include "../datacu.hpp"
#include <cublas_v2.h>
#endif

/* ---------------------------------------------------------------- */
/* BLAS helpers                                                     */
/* ---------------------------------------------------------------- */

namespace vl { namespace impl {

#ifdef APPLE_BLAS
  static inline CBLAS_TRANSPOSE toaAppleTransposeFlag(char c) {
    switch (c) {
      case 'n': case 'N': return CblasNoTrans ;
      case 't': case 'T': return CblasTrans ;
      case 'c': case 'C': return CblasConjTrans ;
      default: assert(false) ;
    }
  }
#endif

template<vl::DeviceType deviceType, vl::DataType dataType>
struct blas
{
  typedef typename DataTypeTraits<dataType>::type type ;

  static vl::ErrorCode
  gemm(vl::Context& context,
       char op1, char op2,
       Int m, Int n, Int k,
       type alpha,
       type const * a, Int lda,
       type const * b, Int ldb,
       type beta,
       type * c, Int ldc) ;

  static vl::ErrorCode
  gemv(vl::Context& context,
       char op,
       Int m, Int n,
       type alpha,
       type const * a, Int lda,
       type const * x, Int incx,
       type beta,
       type * y, Int incy) ;

  static vl::ErrorCode
  axpy(vl::Context& context,
       Int n,
       type alpha,
       type const *x, Int incx,
       type *y, Int incy) ;

  static vl::ErrorCode
  scal(vl::Context& context,
       Int n,
       type alpha,
       type *x,
       Int inc) ;
} ;

/* ---------------------------------------------------------------- */
/* CPU implementation                                               */
/* ---------------------------------------------------------------- */

template<>
struct blas<vl::VLDT_CPU, vl::VLDT_Float>
{
  typedef float type ;

  static vl::ErrorCode
  gemm(vl::Context& context,
       char op1, char op2,
       Int m, Int n, Int k,
       type alpha,
       type const * a, Int lda,
       type const * b, Int ldb,
       type beta,
       type * c, Int ldc)
  {
#ifdef APPLE_BLAS
    cblas_sgemm(CblasColMajor,
                toaAppleTransposeFlag(op1),
                toaAppleTransposeFlag(op2),
                (int)m, (int)n, (int)k,
                alpha,
                (type*)a, (int)lda,
                (type*)b, (int)ldb,
                beta,
                c, (int)ldc) ;
#else
    sgemm(&op1, &op2,
          &m, &n, &k,
          &alpha,
          (type*)a, &lda,
          (type*)b, &ldb,
          &beta,
          c, &ldc) ;
#endif
    return vl::VLE_Success ;
  }

  static vl::ErrorCode
  gemv(vl::Context& context,
       char op,
       Int m, Int n,
       type alpha,
       type const * a, Int lda,
       type const * x, Int incx,
       type beta,
       type * y, Int incy)
  {
#ifdef APPLE_BLAS
    cblas_sgemv(CblasColMajor,
                toaAppleTransposeFlag(op),
                (int)m, (int)n,
                alpha,
                (type*)a, (int)lda,
                (type*)x, (int)incx,
                beta,
                y, (int)incy) ;
#else
    sgemv(&op,
          &m, &n, &alpha,
          (type*)a, &lda,
          (type*)x, &incx,
          &beta,
          y, &incy) ;
#endif
    return vl::VLE_Success ;
  }

  static vl::ErrorCode
  axpy(vl::Context & context,
       Int n,
       type alpha,
       type const *x, Int incx,
       type *y, Int incy)
  {
#if defined(APPLE_BLAS)
    cblas_saxpy((int)n,
                alpha,
                (type*)x, (int)incx,
                (type*)y, (int)incy) ;
#else
    saxpy(&n,
          &alpha,
          (type*)x, &incx,
          (type*)y, &incy) ;
#endif
    return vl::VLE_Success ;
  }

  static vl::ErrorCode
  scal(vl::Context & context,
       Int n,
       type alpha,
       type *x, Int incx)
  {
#if defined(APPLE_BLAS)
    cblas_sscal((int)n,
                alpha,
                (type*)x, (int)incx) ;
#else
    sscal(&n,
          &alpha,
          (type*)x, &incx) ;
#endif
    return vl::VLE_Success ;

  }
} ;

template<>
struct blas<vl::VLDT_CPU, vl::VLDT_Double>
{
  typedef double type ;

  static vl::ErrorCode
  gemm(vl::Context& context,
       char op1, char op2,
       Int m, Int n, Int k,
       type alpha,
       type const * a, Int lda,
       type const * b, Int ldb,
       type beta,
       type * c, Int ldc)
  {
#ifdef APPLE_BLAS
    cblas_dgemm(CblasColMajor,
                toaAppleTransposeFlag(op1),
                toaAppleTransposeFlag(op2),
                (int)m, (int)n, (int)k,
                alpha,
                (type*)a, (int)lda,
                (type*)b, (int)ldb,
                beta,
                c, (int)ldc) ;
#else
    dgemm(&op1, &op2,
          &m, &n, &k,
          &alpha,
          (type*)a, &lda,
          (type*)b, &ldb,
          &beta,
          c, &ldc) ;
#endif
    return vl::VLE_Success ;
  }

  static vl::ErrorCode
  gemv(vl::Context& context,
       char op,
       Int m, Int n,
       type alpha,
       type const * a, Int lda,
       type const * x, Int incx,
       type beta,
       type * y, Int incy)
  {
#ifdef APPLE_BLAS
    cblas_dgemv(CblasColMajor,
                toaAppleTransposeFlag(op),
                (int)m, (int)n,
                alpha,
                (type*)a, (int)lda,
                (type*)x, (int)incx,
                beta,
                y, (int)incy) ;
#else
    dgemv(&op,
          &m, &n, &alpha,
          (type*)a, &lda,
          (type*)x, &incx,
          &beta,
          y, &incy) ;
#endif
    return vl::VLE_Success ;
  }

  static vl::ErrorCode
  axpy(vl::Context & context,
       Int n,
       type alpha,
       type const *x, Int incx,
       type *y, Int incy)
  {
#ifdef APPLE_BLAS
    cblas_daxpy((int)n,
                alpha,
                (type*)x, (int)incx,
                (type*)y, (int)incy) ;
#else
    daxpy(&n,
          &alpha,
          (double*)x, &incx,
          (double*)y, &incy);
#endif
    return vl::VLE_Success ;
  }

  static vl::ErrorCode
  scal(vl::Context & context,
       Int n,
       type alpha,
       type *x, Int incx)
  {
#ifdef APPLE_BLAS
    cblas_dscal((int)n,
                alpha,
                (type*)x, (int)incx) ;
#else
    dscal(&n,
          &alpha,
          (double*)x, &incx) ;
#endif
    return vl::VLE_Success ;
  }
} ;

/* ---------------------------------------------------------------- */
/* GPU implementation                                               */
/* ---------------------------------------------------------------- */

#ifdef ENABLE_GPU

template<>
struct blas<vl::VLDT_GPU, vl::VLDT_Float>
{
  typedef float type ;

  static vl::ErrorCode
  gemm(vl::Context& context,
       char op1, char op2,
       Int m, Int n, Int k,
       type alpha,
       type const * a, Int lda,
       type const * b, Int ldb,
       type beta,
       type * c, Int ldc)
  {
    cublasHandle_t handle ;
    cublasStatus_t status ;
    status = context.getCudaHelper().getCublasHandle(&handle) ;
    if (status != CUBLAS_STATUS_SUCCESS) goto done ;

    status = cublasSgemm(handle,
                         (op1 == 't') ? CUBLAS_OP_T : CUBLAS_OP_N,
                         (op2 == 't') ? CUBLAS_OP_T : CUBLAS_OP_N,
                         (int)m, (int)n, (int)k,
                         &alpha,
                         a, (int)lda,
                         b, (int)ldb,
                         &beta,
                         c, (int)ldc);
  done:
    return context.setError
    (context.getCudaHelper().catchCublasError(status, "cublasSgemm"), __func__) ;
  }

  static vl::ErrorCode
  gemv(vl::Context& context,
       char op,
       Int m, Int n,
       type alpha,
       type const * a, Int lda,
       type const * x, Int incx,
       type beta,
       type * y, Int incy)
  {
    cublasHandle_t handle ;
    cublasStatus_t status ;
    status = context.getCudaHelper().getCublasHandle(&handle) ;
    if (status != CUBLAS_STATUS_SUCCESS) goto done ;
    status = cublasSgemv(handle,
                         (op == 't') ? CUBLAS_OP_T : CUBLAS_OP_N,
                         (int)m, (int)n,
                         &alpha,
                         a, (int)lda,
                         x, (int)incx,
                         &beta,
                         y, (int)incy);

  done:
    return context.setError
      (context.getCudaHelper().catchCublasError(status, "cublasSgemv"), __func__) ;
  }

  static vl::ErrorCode
  scal(vl::Context & context,
       Int n,
       type alpha,
       type *x, Int incx)
  {
    cublasHandle_t handle ;
    cublasStatus_t status ;
    status = context.getCudaHelper().getCublasHandle(&handle) ;
    if (status != CUBLAS_STATUS_SUCCESS) goto done ;
    status = cublasSscal(handle,
                         (int)n,
                         &alpha,
                         x, (int)incx) ;
  done:
    return context.setError
      (context.getCudaHelper().catchCublasError(status, "cublasSscal"), __func__) ;
  }

  static vl::ErrorCode
  axpy(vl::Context & context,
       Int n,
       type alpha,
       type const *x, Int incx,
       type *y, Int incy)
  {
    cublasHandle_t handle ;
    cublasStatus_t status ;
    status = context.getCudaHelper().getCublasHandle(&handle) ;
    if (status != CUBLAS_STATUS_SUCCESS) goto done ;
    status = cublasSaxpy(handle,
                         (int)n,
                         &alpha,
                         x, (int)incx,
                         y, (int)incy) ;
  done:
    return context.setError
      (context.getCudaHelper().catchCublasError(status, "cublasSaxpy"), __func__) ;
  }
} ;

template<>
struct blas<vl::VLDT_GPU, vl::VLDT_Double>
{
  typedef double type ;

  static vl::ErrorCode
  gemm(vl::Context& context,
       char op1, char op2,
       Int m, Int n, Int k,
       type alpha,
       type const * a, Int lda,
       type const * b, Int ldb,
       type beta,
       type * c, Int ldc)
  {
    cublasHandle_t handle ;
    cublasStatus_t status ;
    status = context.getCudaHelper().getCublasHandle(&handle) ;
    if (status != CUBLAS_STATUS_SUCCESS) goto done ;

    status = cublasDgemm(handle,
                         (op1 == 't') ? CUBLAS_OP_T : CUBLAS_OP_N,
                         (op2 == 't') ? CUBLAS_OP_T : CUBLAS_OP_N,
                         (int)m, (int)n, (int)k,
                         &alpha,
                         a, (int)lda,
                         b, (int)ldb,
                         &beta,
                         c, (int)ldc);
  done:
    return context.setError
    (context.getCudaHelper().catchCublasError(status, "cublasDgemm"), __func__) ;
  }

  static vl::ErrorCode
  gemv(vl::Context& context,
       char op,
       Int m, Int n,
       type alpha,
       type const * a, Int lda,
       type const * x, Int incx,
       type beta,
       type * y, Int incy)
  {
    cublasHandle_t handle ;
    cublasStatus_t status ;
    status = context.getCudaHelper().getCublasHandle(&handle) ;
    if (status != CUBLAS_STATUS_SUCCESS) goto done ;

    status = cublasDgemv(handle,
                         (op == 't') ? CUBLAS_OP_T : CUBLAS_OP_N,
                         (int)m, (int)n,
                         &alpha,
                         a, (int)lda,
                         x, (int)incx,
                         &beta,
                         y, (int)incy);
  done:
    return context.setError
    (context.getCudaHelper().catchCublasError(status, "cublasDgemv"), __func__) ;
  }

  static vl::ErrorCode
  scal(vl::Context & context,
       Int n,
       type alpha,
       type *x, Int incx)
  {
    cublasHandle_t handle ;
    cublasStatus_t status ;
    status = context.getCudaHelper().getCublasHandle(&handle) ;
    if (status != CUBLAS_STATUS_SUCCESS) goto done ;
    status = cublasDscal(handle,
                         (int)n,
                         &alpha,
                         x, (int)incx) ;
  done:
    return context.setError
      (context.getCudaHelper().catchCublasError(status, "cublasDscal"), __func__) ;
  }

  static vl::ErrorCode
  axpy(vl::Context & context,
       Int n,
       type alpha,
       type const *x, Int incx,
       type *y, Int incy)
  {
    cublasHandle_t handle ;
    cublasStatus_t status ;
    status = context.getCudaHelper().getCublasHandle(&handle) ;
    if (status != CUBLAS_STATUS_SUCCESS) goto done ;
    status = cublasDaxpy(handle,
                         (int)n,
                         &alpha,
                         x, (int)incx,
                         y, (int)incy) ;
  done:
    return context.setError
      (context.getCudaHelper().catchCublasError(status, "cublasDaxpy"), __func__) ;
  }
} ;
#endif // ENABLE_GPU

} } // namespace vl { namespace impl {
#endif /* defined(__vl__blashelper__) */
