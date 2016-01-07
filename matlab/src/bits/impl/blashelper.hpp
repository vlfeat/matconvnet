// @file blashelper.hpp
// @brief BLAS helpers
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__blashelper__
#define __vl__blashelper__

#include "../data.hpp"

#include <blas.h>
#ifdef ENABLE_GPU
#include "../datacu.hpp"
#include <cublas_v2.h>
#endif

/* ---------------------------------------------------------------- */
/* BLAS helpers                                                     */
/* ---------------------------------------------------------------- */

namespace vl { namespace impl {

template<vl::Device deviceType, vl::Type dataType>
struct blas
{
  typedef typename DataTypeTraits<dataType>::type type ;

  static vl::Error
  gemm(vl::Context& context,
       char op1, char op2,
       ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
       type alpha,
       type const * a, ptrdiff_t lda,
       type const * b, ptrdiff_t ldb,
       type beta,
       type * c, ptrdiff_t ldc) ;

  static vl::Error
  gemv(vl::Context& context,
       char op,
       ptrdiff_t m, ptrdiff_t n,
       type alpha,
       type const * a, ptrdiff_t lda,
       type const * x, ptrdiff_t incx,
       type beta,
       type * y, ptrdiff_t incy) ;

} ;

/* ---------------------------------------------------------------- */
/* CPU implementation                                               */
/* ---------------------------------------------------------------- */

template<>
struct blas<vl::CPU, vl::vlTypeFloat>
{
  typedef float type ;

  static vl::Error
  gemm(vl::Context& context,
       char op1, char op2,
       ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
       type alpha,
       type const * a, ptrdiff_t lda,
       type const * b, ptrdiff_t ldb,
       type beta,
       type * c, ptrdiff_t ldc)
  {
    sgemm(&op1, &op2,
          &m, &n, &k,
          &alpha,
          (type*)a, &lda,
          (type*)b, &ldb,
          &beta,
          c, &ldc) ;
    return vl::vlSuccess ;
  }

  static vl::Error
  gemv(vl::Context& context,
       char op,
       ptrdiff_t m, ptrdiff_t n,
       type alpha,
       type const * a, ptrdiff_t lda,
       type const * x, ptrdiff_t incx,
       type beta,
       type * y, ptrdiff_t incy)
  {
    sgemv(&op,
          &m, &n, &alpha,
          (float*)a, &lda,
          (float*)x, &incx,
          &beta,
          y, &incy) ;
    return vl::vlSuccess ;
  }
} ;

template<>
struct blas<vl::CPU, vl::vlTypeDouble>
{
  typedef double type ;

  static vl::Error
  gemm(vl::Context& context,
       char op1, char op2,
       ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
       type alpha,
       type const * a, ptrdiff_t lda,
       type const * b, ptrdiff_t ldb,
       type beta,
       type * c, ptrdiff_t ldc)
  {
    dgemm(&op1, &op2,
          &m, &n, &k,
          &alpha,
          (type*)a, &lda,
          (type*)b, &ldb,
          &beta,
          c, &ldc) ;
    return vl::vlSuccess ;
  }

  static vl::Error
  gemv(vl::Context& context,
       char op,
       ptrdiff_t m, ptrdiff_t n,
       type alpha,
       type const * a, ptrdiff_t lda,
       type const * x, ptrdiff_t incx,
       type beta,
       type * y, ptrdiff_t incy)
  {
    dgemv(&op,
          &m, &n, &alpha,
          (type*)a, &lda,
          (type*)x, &incx,
          &beta,
          y, &incy) ;
    return vl::vlSuccess ;
  }
} ;

/* ---------------------------------------------------------------- */
/* GPU implementation                                               */
/* ---------------------------------------------------------------- */

#ifdef ENABLE_GPU

template<>
struct blas<vl::GPU, vl::vlTypeFloat>
{
  typedef float type ;

  static vl::Error
  gemm(vl::Context& context,
       char op1, char op2,
       ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
       type alpha,
       type const * a, ptrdiff_t lda,
       type const * b, ptrdiff_t ldb,
       type beta,
       type * c, ptrdiff_t ldc)
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

  static vl::Error
  gemv(vl::Context& context,
       char op,
       ptrdiff_t m, ptrdiff_t n,
       type alpha,
       type const * a, ptrdiff_t lda,
       type const * x, ptrdiff_t incx,
       type beta,
       type * y, ptrdiff_t incy)
  {
    cublasHandle_t handle ;
    cublasStatus_t status ;
    status = context.getCudaHelper().getCublasHandle(&handle) ;
    if (status != CUBLAS_STATUS_SUCCESS) goto done ;

    status = cublasSgemv(handle,
                         (op == 't') ? CUBLAS_OP_T : CUBLAS_OP_N,
                         (int)m, (int)n,
                         &alpha,
                         a, lda,
                         x, (int)incx,
                         &beta,
                         y, (int)incy);

  done:
    return context.setError
    (context.getCudaHelper().catchCublasError(status, "cublasSgemv"), __func__) ;
  }
} ;

template<>
struct blas<vl::GPU, vl::vlTypeDouble>
{
  typedef double type ;

  static vl::Error
  gemm(vl::Context& context,
       char op1, char op2,
       ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
       type alpha,
       type const * a, ptrdiff_t lda,
       type const * b, ptrdiff_t ldb,
       type beta,
       type * c, ptrdiff_t ldc)
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

  static vl::Error
  gemv(vl::Context& context,
       char op,
       ptrdiff_t m, ptrdiff_t n,
       type alpha,
       type const * a, ptrdiff_t lda,
       type const * x, ptrdiff_t incx,
       type beta,
       type * y, ptrdiff_t incy)
  {
    cublasHandle_t handle ;
    cublasStatus_t status ;
    status = context.getCudaHelper().getCublasHandle(&handle) ;
    if (status != CUBLAS_STATUS_SUCCESS) goto done ;

    status = cublasDgemv(handle,
                         (op == 't') ? CUBLAS_OP_T : CUBLAS_OP_N,
                         (int)m, (int)n,
                         &alpha,
                         a, lda,
                         x, (int)incx,
                         &beta,
                         y, (int)incy);
  done:
    return context.setError
    (context.getCudaHelper().catchCublasError(status, "cublasDgemv"), __func__) ;
  }
} ;
#endif // ENABLE_GPU
} } // namespace vl { namespace impl {
#endif /* defined(__vl__blashelper__) */
