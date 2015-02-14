// @file blashelper.hpp
// @brief BLAS helpers
// @author Andrea Vedaldi

/*
Copyright (C) 2015 Andrea Vedaldi.
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
/* Type helper                                                      */
/* ---------------------------------------------------------------- */

template <typename type>
struct get_vl_type { operator vl::Type() const ; } ;

template <>
struct get_vl_type<float> { operator vl::Type() const { return vl::vlTypeFloat ; } } ;

template <>
struct get_vl_type<double> { operator vl::Type() const { return vl::vlTypeDouble ; } } ;

/* ---------------------------------------------------------------- */
/* GEMM helper                                                      */
/* ---------------------------------------------------------------- */

template<vl::Device arch, typename type> vl::Error
gemm(vl::Context& context,
          char op1, char op2,
     ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
     type alpha,
     type const * a, ptrdiff_t lda,
     type const * b, ptrdiff_t ldb,
     type beta,
          type * c, ptrdiff_t ldc) ;

template<> inline vl::Error
gemm<vl::CPU, float>(vl::Context& context,
                     char op1, char op2,
                     ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
                     float alpha,
                     float const * a, ptrdiff_t lda,
                     float const * b, ptrdiff_t ldb,
                     float beta,
                     float * c, ptrdiff_t ldc)
{
  sgemm(&op1, &op2,
        &m, &n, &k,
        &alpha,
        (float*)a, &lda,
        (float*)b, &ldb,
        &beta,
        c, &ldc) ;
  return vl::vlSuccess ;
}

#ifdef ENABLE_GPU
template<> inline vl::Error
gemm<vl::GPU, float>(vl::Context& context,
                     char op1, char op2,
                     ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
                     float alpha,
                     float const * a, ptrdiff_t lda,
                     float const * b, ptrdiff_t ldb,
                     float beta,
                     float * c, ptrdiff_t ldc)
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
  (context.getCudaHelper().catchCublasError(status, "cublasSgemm"), "gemm<>: ") ;
}
#endif

/* ---------------------------------------------------------------- */
/* GEMV helper                                                      */
/* ---------------------------------------------------------------- */

template<vl::Device arch, typename type> vl::Error
gemv(vl::Context& context,
          char op,
          ptrdiff_t m, ptrdiff_t n,
          type alpha,
          type const * a, ptrdiff_t lda,
          type const * x, ptrdiff_t incx,
          type beta,
          type * y, ptrdiff_t incy) ;

template<> inline vl::Error
gemv<vl::CPU, float>(vl::Context& context,
                     char op,
                     ptrdiff_t m, ptrdiff_t n,
                     float alpha,
                     float const * a, ptrdiff_t lda,
                     float const * x, ptrdiff_t incx,
                     float beta,
                     float * y, ptrdiff_t incy)
{
  sgemv(&op,
        &m, &n, &alpha,
        (float*)a, &lda,
        (float*)x, &incx,
        &beta,
        y, &incy) ;
  return vl::vlSuccess ;
}

#ifdef ENABLE_GPU
template<> inline vl::Error
gemv<vl::GPU, float>(vl::Context& context,
                     char op,
                     ptrdiff_t m, ptrdiff_t n,
                     float alpha,
                     float const * a, ptrdiff_t lda,
                     float const * x, ptrdiff_t incx,
                     float beta,
                     float * y, ptrdiff_t incy)
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
                       y, (int)incy) ;
done:
  return context.setError
  (context.getCudaHelper().catchCublasError(status, "cublasSgemv"), "gemv<>: ") ;
}
#endif

#endif /* defined(__vl__blashelper__) */
