// @file nnnormalize.cu
// @brief Normalization block
// @author Andrea Vedaldi

/*
Copyright (C) 2014-17 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnnormalize.hpp"
#include "impl/dispatcher.hpp"
#include <cmath>
#include <cassert>
#include <cstring>

using namespace vl ;
using namespace vl::nn ;
using namespace vl::impl ;

template<vl::DeviceType deviceType, vl::DataType dataType> struct LRNForward ;
template<vl::DeviceType deviceType, vl::DataType dataType> struct LRNBackward ;

// -------------------------------------------------------------------
//                                Fast approximated numerical routines
// -------------------------------------------------------------------

#ifndef _MSC_VER
#include <x86intrin.h>
#pragma GCC optimize ("fast-math")
#pragma GCC optimize ("tree-vectorize")
//#pragma GCC target ("veclibabi=svml")
//#pragma GCC target "sse4"
#endif
#define restrict __restrict

#define VL_NNNORMALIZE_FAST
#define max(a,b) (((a)>=(b))?a:b)
#define xat(t) x[(t) * offset]
#define yat(t) y[(t) * offset]
#define zat(t) z[(t) * offset]

#ifndef VL_NNNORMALIZE_FAST
inline double fast_pow(double a, double b) { return pow(a,b) ; }
inline float fast_pow(float a, float b) { return powf(a,b) ; }
#else
//#define VERY_FAST
#ifndef VERY_FAST
inline double fast_pow(double x, double y)
{
  double z ;
  double const plog3 = 0.164042561333445 ;
  double const plog2 = -0.606737602222409 ;
  double const plog1 = 1.442695040888963 ;
  double const pexp3 = 0.079441541679836 ;
  double const pexp2 = 0.227411277760219 ;
  double const pexp1 = 0.693147180559945 ;
  typedef long long int int_t;
  const int_t offset = 1023LL << 52 ;

  int_t ix = *(int_t*)&x - offset ;
  int_t imx = (ix & ((1LL<<52)-1LL)) + offset;
  double fx = (double)(ix >> 52) ;
  double mx = *((double*)&imx) - 1 ;
  double mx2 = mx*mx ;
  double mx3 = mx2*mx ;
  double t = y * (fx + mx*plog1 + mx2*plog2 + mx3*plog3) ;
  //  double t = y * (fx + mx) ;

  double fz = floor(t) ;
  double rz = t - fz ;
  double rz2 = rz*rz ;
  double rz3 = rz2*rz ;
  double tz = fz + rz*pexp1 + rz2*pexp2 + rz3*pexp3 ;
  // double tz = fz + rz ;

  //  mexPrintf("%g %g -- ix %ld imx %ld fx %g mx %g t %g\n", x,y, ix,imx, fx, mx, t) ;
  *((int_t*)&z) = (int_t)(tz * (1LL<<52)) + offset ;
  //z = exp(t * log(2.0)) ;
  return z ;
}
#else
inline double fast_pow(double a, double b)
{
  double z ;
  typedef long long int int_t;
  const int_t offset = 1023L << 52 ;
  int_t ai = *((int_t*)&a) ;
  *((int_t*)&z) = (int_t)(b * (ai - offset)) + offset ;
  return z ;
}
#endif

#ifndef VERY_FAST
inline float fast_pow(float x, float y)
{
  float z ;
  float const plog3 = 0.164042561333445F ;
  float const plog2 = -0.606737602222409F ;
  float const plog1 = 1.442695040888963F ;
  float const pexp3 = 0.079441541679836F ;
  float const pexp2 = 0.227411277760219F ;
  float const pexp1 = 0.693147180559945F ;
  typedef int int_t;
  const int_t offset = 127 << 23 ;

  int_t ix = *(int_t*)&x - offset ;
  int_t imx = (ix & ((1<<23)-1)) + offset;
  float fx = (float)(ix >> 23) ;
  float mx = *((float*)&imx) - 1 ;
  float mx2 = mx*mx ;
  float mx3 = mx2*mx ;
  float t = y * (fx + mx*plog1 + mx2*plog2 + mx3*plog3) ;

  float fz = floor(t) ;
  float rz = t - fz ;
  float rz2 = rz*rz ;
  float rz3 = rz2*rz ;
  float tz = fz + rz*pexp1 + rz2*pexp2 + rz3*pexp3 ;

  *((int_t*)&z) = (int_t)(tz * (1<<23)) + offset ;
  return z ;
}
#else
inline float fast_pow(float a, float b)
{
  float z ;
  typedef int int_t;
  const int_t offset = 127 << 23 ;
  int_t ai = *((int_t*)&a) ;
  *((int_t*)&z) = (int_t)(b * (ai - offset)) + offset ;
  return z ;
}
#endif
#endif

// -------------------------------------------------------------------
//                                                         Forward CPU
// -------------------------------------------------------------------

template<vl::DataType dataType>
struct LRNForward<vl::VLDT_CPU, dataType>
{
  vl::ErrorCode operator()(vl::nn::LRN &op,
                           vl::Tensor &output,
                           vl::Tensor const &input)
  {
    typedef typename vl::DataTypeTraits<dataType>::type type ;
    auto width = output.getWidth() ;
    auto height = output.getHeight() ;
    auto depth = output.getDepth() ;
    auto num = output.getSize() ;
    auto inputData = (type const*)input.getMemory() ;
    auto outputData = (type*)output.getMemory() ;

    int t ;
    int m1 = ((signed)op.normDepth-1)/2 ;
    int m2 = (int)op.normDepth - m1 - 1 ;
    int offset = (int)width*(int)height ;
#ifndef VL_NNNORMALIZE_FAST
    for (int k = 0 ; k < num ; ++k) {
      for (int h = 0 ; h < height ; ++h) {
        for (int w = 0 ; w < width ; ++w) {
          type const* x = data + w + h * width ;
          type* y = output + w + h * width ;
          type acc = 0 ;
          for (t = -m2 ; t < (signed)depth ; ++t) {
            type ap = 0 ;
            type am = 0 ;
            if (t-m1-1 >= 0) { am = xat(t-m1-1) ; }
            if (t+m2 < depth) { ap = xat(t+m2) ; }
            acc += ap*ap - am*am ;
            if (0 <= t && t < depth) {
              yat(t) = xat(t) * fast_pow(kappa + alpha * acc, -beta) ;
            }
          }
        }
      }
      data += width*height*depth ;
      output += width*height*depth ;
    }
#else
    type * acc = (type*) calloc(sizeof(type), width*height) ;
    for (int k = 0 ; k < num ; ++k) {
      memset(acc, 0, sizeof(type) * width*height) ;
      for (t = -m2 ; t < (signed)depth ; ++t) {
        int tm = t - m1 - 1 ;
        int tp = t + m2 ;
        type const* xam = inputData + offset * (t-m1-1) ;
        type const* xap = inputData + offset * (t+m2) ;
        type *end = acc + width*height ;
        if (0 <= tm && tp < depth) {
          for(type *xacc = acc ; xacc != end ; ++xacc, ++xam, ++xap) {
            type am = *xam ;
            type ap = *xap ;
            *xacc += ap*ap - am*am ;
          }
        } else if (0 > tm && tp < depth) {
          for(type *xacc = acc ; xacc != end ; ++xacc, ++xap) {
            type ap = *xap ;
            *xacc += ap*ap ;
          }
        } else if (0 <= tm && tp >= depth) {
          for(type *xacc = acc ; xacc != end ; ++xacc, ++xam) {
            type am = *xam ;
            *xacc -= am*am ;
          }
        }
        if (0 <= t && t < depth) {
          type const* xx = inputData + offset * t ;
          type * xy = outputData + offset * t ;
          for(type *xacc = acc ; xacc != end ; ++xacc, ++xx, ++xy) {
            (*xy) = (*xx) * fast_pow(op.kappa + op.alpha * (*xacc), -op.beta) ;
          }
        }
      }
      inputData += width*height*depth ;
      outputData += width*height*depth ;
    }
    free(acc) ;
#endif
    return VLE_Success ;
  }
} ;

// -------------------------------------------------------------------
//                                                        Backward CPU
// -------------------------------------------------------------------

template<vl::DataType dataType>
struct LRNBackward<vl::VLDT_CPU, dataType>
{
  vl::ErrorCode operator()(vl::nn::LRN &op,
                           vl::Tensor &derInput,
                           vl::Tensor const &input,
                           vl::Tensor const &derOutput)
  {
    typedef typename vl::DataTypeTraits<dataType>::type type ;
    auto width = derOutput.getWidth() ;
    auto height = derOutput.getHeight() ;
    auto depth = derOutput.getDepth() ;
    auto num = derOutput.getSize() ;
    auto inputData = (type const*)input.getMemory() ;
    auto derOutputData = (type const*)derOutput.getMemory() ;
    auto derInputData = (type*)derInput.getMemory() ;

    int m1 = ((signed)op.normDepth-1)/2 ;
    int m2 = (int)op.normDepth - m1 - 1 ;
    int offset = (int)width*(int)height ;
    type ab2 = 2*op.alpha*op.beta ;
    int t ;

#ifndef VL_NNNORMALIZE_FAST
    int q ;
    for (int k = 0 ; k < num ; ++k) {
      for (int h = 0 ; h < height ; ++h) {
        for (int w = 0 ; w < width ; ++w) {
          type const* x = data + w + h * width ;
          T* y = output + w + h * width ;
          type const* z = derOutput + w + h * width ;
          type acc = 0 ;
          for (t = 0 ; t < (signed)depth ; ++t) {
            yat(t) = 0 ;
          }
          for (t = -m2 ; t < (signed)depth ; ++t) {
            int q1 = t-m1 ;
            int q2 = t+m2 ;
            type ap = 0 ;
            type am = 0 ;
            if (t-m1-1 >= 0) { am = xat(t-m1-1) ; } else { q1 = 0 ; }
            if (t+m2 < depth) { ap = xat(t+m2) ; } else { q2 = depth - 1 ; }
            acc += ap*ap - am*am ;
            type L = kappa + alpha * acc ;
            type Lbeta = fast_pow(L, -beta) ;
            type Lbeta1 = Lbeta / L ;

            if (0 <= t && t < depth) {
              yat(t) += zat(t) * Lbeta ;
              for (q = q1 ; q <= q2 ; ++ q) {
                yat(q) -= zat(t) * xat(t) * xat(q) * ab2 * Lbeta1 ;
              }
            }
          }
        }
      }
      data += width*height*depth ;
      output += width*height*depth ;
      derOutput += width*height*depth ;
    }
#else
    type * restrict acc = (type*) malloc(sizeof(type) * width*height) ;
    type * restrict acc2 = (type*) malloc(sizeof(type) * width*height*depth) ;
    for (int k = 0 ; k < num ; ++k) {
      memset(acc, 0, sizeof(type) * width*height) ;
      for (t = -m2 ; t < (signed)depth ; ++t) {
        /*
         Compue the square of the input data x.^2 summed in the normalization window. This is done
         incrementally, by updating the previous normalization window sum.
         */
        {
          int const tm = t - m1 - 1 ;
          int const tp = t + m2 ;
          type const* restrict datam_ = inputData + offset * tm ;
          type const* restrict datap_ = inputData + offset * tp ;
          type *end = acc + width*height ;

          if (0 <= tm && tp < depth) {
            for(type * restrict acc_ = acc ; acc_ != end ; ++acc_, ++datap_, ++datam_) {
              type am = *datam_ ;
              type ap = *datap_ ;
              *acc_ += ap*ap - am*am ;
            }
          } else if (0 > tm && tp < depth) {
            for(type * restrict acc_ = acc ; acc_ != end ; ++acc_, ++datap_) {
              type ap = *datap_ ;
              *acc_ += ap*ap ;
            }
          } else if (0 <= tm && tp >= depth) {
            for(type * restrict acc_ = acc ; acc_ != end ; ++acc_, ++datam_) {
              type am = *datam_ ;
              *acc_ -= am*am ;
            }
          }
        }

        /*
         Compute the arguments of the summation in the derivative
         expression, storing them into acc2.
         */
        if (0 <= t && t < depth) {
          type const* restrict data_ = inputData + offset * t ;
          type const* restrict derOutput_ = derOutputData + offset * t ;
          type * restrict output_ = derInputData + offset * t ;
          type * restrict acc2_ = acc2 + offset * t ;
          type * end = acc + width*height ;
          for(type * restrict acc_ = acc ; acc_ != end ;
              ++acc_, ++acc2_, ++data_, ++derOutput_, ++output_) {
            type L = op.kappa + op.alpha * (*acc_) ;
            type Lbeta = fast_pow(L, -(type)op.beta) ;
            type temp1 = (*derOutput_) * Lbeta ;
            type temp2 = (*data_) * ab2 * temp1 / L ;
            *output_ = temp1 ;
            *acc2_ = temp2 ;
          }
        }
      }

      /*
       Integrate along feature channels in acc2, summing plane t-1 to
       plane t.
       */
      for (t = 1 ; t < (signed)depth ; ++t) {
        type * restrict acc2_ = acc2 + t * offset ;
        type const* restrict src_ = acc2_ - offset ;
        type const* end = acc2_ + offset ;
        for( ; acc2_ != end ; ++acc2_, ++src_) {
          *acc2_ += *src_ ;
        }
      }

      /*
       Compute summation in the derivative expression from the integral
       just obtained.
       */
      for (t = 0 ; t < (signed)depth ; ++t) {
        int q1 = t - m2 - 1 ;
        int q2 = ((t + m1) <= (depth - 1)) ? t + m1 : depth - 1 ;
        type const* restrict acc22_ = acc2 + offset * q2 ;
        type const* restrict acc21_ = acc2 + offset * q1 ;
        type const* restrict data_  = inputData + offset * t ;
        type const* restrict end = data_  + width*height ;
        type * restrict output_ = derInputData + offset * t ;
        if (q1 >= 0) {
          for( ; data_ != end ; ++data_, ++acc22_, ++acc21_, ++output_) {
            *output_ -= (*acc22_ - *acc21_) * (*data_) ;
          }
        } else {
          for( ; data_ != end ; ++data_, ++acc22_, ++output_) {
            *output_ -= (*acc22_) * (*data_) ;
          }
        }
      }
      inputData += width*height*depth ;
      derInputData += width*height*depth ;
      derOutputData += width*height*depth ;
    }
    free(acc) ;
    free(acc2) ;
#endif
    return VLE_Success ;
  }
} ;

/* ---------------------------------------------------------------- */
/*                                                           Driver */
/* ---------------------------------------------------------------- */

#if ENABLE_GPU
#include "nnnormalize_gpu.cu"
#endif

LRN::LRN(vl::Context &context,
         int normDepth,
         double kappa,
         double alpha,
         double beta)
: context(context), normDepth(normDepth), kappa(kappa), alpha(alpha), beta(beta)
{ }

vl::ErrorCode
LRN::forward(vl::Tensor &output,
             vl::Tensor const &input)
{
  return dispatch<LRNForward>()(*this,output,input) ;
}

vl::ErrorCode
LRN::backward(vl::Tensor &derInput,
              vl::Tensor const &input,
              vl::Tensor const &derOutput)
{
  return dispatch<LRNBackward>()(*this,derInput,input,derOutput) ;
}
