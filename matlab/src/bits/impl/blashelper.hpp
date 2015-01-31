/** @file blashelper.hpp
 ** @brief Helper functions to call BLAS and cuBLAS code
 ** @author Andrea Vedaldi
 **/

#ifndef __matconv__blashelper__
#define __matconv__blashelper__

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
struct get_type_id { operator vl::Type() const ; } ;

template <>
struct get_type_id<float> { operator vl::Type() const { return vl::FLOAT ; } } ;

template <>
struct get_type_id<double> { operator vl::Type() const { return vl::DOUBLE ; } } ;

/* ---------------------------------------------------------------- */
/* GEMM helper                                                      */
/* ---------------------------------------------------------------- */

template<vl::Device arch, typename type>
void gemm(vl::Context& context,
          char op1, char op2,
          ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
          type alpha,
          type const * a, ptrdiff_t lda,
          type const * b, ptrdiff_t ldb,
          type beta,
          type * c, ptrdiff_t ldc) ;

template<> inline void
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
}

#ifdef ENABLE_GPU
template<> inline void
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
  context.getCudaHelper().getCuBLASHandle(&handle) ;
  cublasSgemm(handle,
              (op1 == 't') ? CUBLAS_OP_T : CUBLAS_OP_N,
              (op2 == 't') ? CUBLAS_OP_T : CUBLAS_OP_N,
              (int)m, (int)n, (int)k,
              &alpha,
              a, (int)lda,
              b, (int)ldb,
              &beta,
              c, (int)ldc);
}
#endif

/* ---------------------------------------------------------------- */
/* GEMV helper                                                      */
/* ---------------------------------------------------------------- */

template<vl::Device arch, typename type>
void gemv(vl::Context& context,
          char op,
          ptrdiff_t m, ptrdiff_t n,
          type alpha,
          type const * a, ptrdiff_t lda,
          type const * x, ptrdiff_t incx,
          type beta,
          type * y, ptrdiff_t incy) ;

template<> inline void
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
}

#ifdef ENABLE_GPU
template<> inline void
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
  context.getCudaHelper().getCuBLASHandle(&handle) ;
  cublasSgemv(handle,
              (op == 't') ? CUBLAS_OP_T : CUBLAS_OP_N,
              (int)m, (int)n,
              &alpha,
              a, lda,
              x, (int)incx,
              &beta,
              y, (int)incy) ;
}
#endif

#endif /* defined(__matconv__blashelper__) */
