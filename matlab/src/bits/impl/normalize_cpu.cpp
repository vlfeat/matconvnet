// @file normalize_cpu.cpp
// @brief Normalize block implementation (CPU)
// @author Andrea Vedaldi

/*
Copyright (C) 2014-15 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "normalize.hpp"
#include "../data.hpp"
#include <math.h>
#include <memory.h>

/* ---------------------------------------------------------------- */
/*                             Fast approximated numerical routines */
/* ---------------------------------------------------------------- */

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

/* ---------------------------------------------------------------- */
/*                                                normalize_forward */
/* ---------------------------------------------------------------- */

template<typename T> static inline void
normalize_forward_cpu(T* normalized,
                      T const* data,
                      size_t width,
                      size_t height,
                      size_t depth,
                      size_t num,
                      size_t normDepth,
                      T kappa, T alpha, T beta)
{
  int t ;
  int m1 = ((signed)normDepth-1)/2 ;
  int m2 = (int)normDepth - m1 - 1 ;
  int offset = (int)width*(int)height ;
#ifndef VL_NNNORMALIZE_FAST
  for (int k = 0 ; k < num ; ++k) {
    for (int h = 0 ; h < height ; ++h) {
      for (int w = 0 ; w < width ; ++w) {
        T const* x = data + w + h * width ;
        T* y = normalized + w + h * width ;
        T acc = 0 ;
        for (t = -m2 ; t < (signed)depth ; ++t) {
          T ap = 0 ;
          T am = 0 ;
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
    normalized += width*height*depth ;
  }
#else
  T * acc = (T*) calloc(sizeof(T), width*height) ;
  for (int k = 0 ; k < num ; ++k) {
    memset(acc, 0, sizeof(T) * width*height) ;
    for (t = -m2 ; t < (signed)depth ; ++t) {
      int tm = t - m1 - 1 ;
      int tp = t + m2 ;
      T const* xam = data + offset * (t-m1-1) ;
      T const* xap = data + offset * (t+m2) ;
      T *end = acc + width*height ;
      if (0 <= tm && tp < depth) {
        for(T *xacc = acc ; xacc != end ; ++xacc, ++xam, ++xap) {
          T am = *xam ;
          T ap = *xap ;
          *xacc += ap*ap - am*am ;
        }
      } else if (0 > tm && tp < depth) {
        for(T *xacc = acc ; xacc != end ; ++xacc, ++xap) {
          T ap = *xap ;
          *xacc += ap*ap ;
        }
      } else if (0 <= tm && tp >= depth) {
        for(T *xacc = acc ; xacc != end ; ++xacc, ++xam) {
          T am = *xam ;
          *xacc -= am*am ;
        }
      }
      if (0 <= t && t < depth) {
        T const* xx = data + offset * t ;
        T* xy = normalized + offset * t ;
        for(T *xacc = acc ; xacc != end ; ++xacc, ++xx, ++xy) {
          (*xy) = (*xx) * fast_pow(kappa + alpha * (*xacc), -beta) ;
        }
      }
    }
    data += width*height*depth ;
    normalized += width*height*depth ;
  }
  free(acc) ;
#endif
}

template<> vl::Error
vl::impl::normalize_forward<vl::CPU, float>
(float* normalized,
 float const* data,
 size_t height, size_t width, size_t depth, size_t size,
 size_t normDepth,
 double kappa, double alpha, double beta)
{
  normalize_forward_cpu<float>(normalized,data,
                               height,width,depth,size,
                               normDepth,kappa,alpha,beta) ;
  return vlSuccess ;
}

/* ---------------------------------------------------------------- */
/*                                               normalize_backward */
/* ---------------------------------------------------------------- */

template<typename T> static inline void
normalize_backward_cpu(T* normalized,
                       T const* data,
                       T const* dzdy,
                       size_t width,
                       size_t height,
                       size_t depth,
                       size_t num,
                       size_t normDepth,
                       T kappa, T alpha, T beta)
{
  int m1 = ((signed)normDepth-1)/2 ;
  int m2 = (int)normDepth - m1 - 1 ;
  int offset = (int)width*(int)height ;
  T ab2 = 2*alpha*beta ;
  int t, q ;

#ifndef VL_NNNORMALIZE_FAST
  for (int k = 0 ; k < num ; ++k) {
    for (int h = 0 ; h < height ; ++h) {
      for (int w = 0 ; w < width ; ++w) {
        T const* x = data + w + h * width ;
        T* y = normalized + w + h * width ;
        T const* z = dzdy + w + h * width ;
        T acc = 0 ;
        for (t = 0 ; t < (signed)depth ; ++t) {
          yat(t) = 0 ;
        }
        for (t = -m2 ; t < (signed)depth ; ++t) {
          int q1 = t-m1 ;
          int q2 = t+m2 ;
          T ap = 0 ;
          T am = 0 ;
          if (t-m1-1 >= 0) { am = xat(t-m1-1) ; } else { q1 = 0 ; }
          if (t+m2 < depth) { ap = xat(t+m2) ; } else { q2 = depth - 1 ; }
          acc += ap*ap - am*am ;
          T L = kappa + alpha * acc ;
          T Lbeta = fast_pow(L, -beta) ;
          T Lbeta1 = Lbeta / L ;

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
    normalized += width*height*depth ;
    dzdy += width*height*depth ;
  }
#else
  T * restrict acc = (T*) malloc(sizeof(T) * width*height) ;
  T * restrict acc2 = (T*) malloc(sizeof(T) * width*height*depth) ;
  for (int k = 0 ; k < num ; ++k) {
    memset(acc, 0, sizeof(T) * width*height) ;
    for (t = -m2 ; t < (signed)depth ; ++t) {
      /*
       Compue the square of the input data x.^2 summed in the normalization window. This is done
       incrementally, by updating the previous normalization window sum.
       */
      {
        int const tm = t - m1 - 1 ;
        int const tp = t + m2 ;
        T const* restrict datam_ = data + offset * tm ;
        T const* restrict datap_ = data + offset * tp ;
        T *end = acc + width*height ;

        if (0 <= tm && tp < depth) {
          for(T * restrict acc_ = acc ; acc_ != end ; ++acc_, ++datap_, ++datam_) {
            T am = *datam_ ;
            T ap = *datap_ ;
            *acc_ += ap*ap - am*am ;
          }
        } else if (0 > tm && tp < depth) {
          for(T * restrict acc_ = acc ; acc_ != end ; ++acc_, ++datap_) {
            T ap = *datap_ ;
            *acc_ += ap*ap ;
          }
        } else if (0 <= tm && tp >= depth) {
          for(T * restrict acc_ = acc ; acc_ != end ; ++acc_, ++datam_) {
            T am = *datam_ ;
            *acc_ -= am*am ;
          }
        }
      }

      /*
       Compute the arguments of the summation in the derivative
       expression, storing them into acc2.
       */
      if (0 <= t && t < depth) {
        T const* restrict data_ = data + offset * t ;
        T const* restrict dzdy_ = dzdy + offset * t ;
        T * restrict normalized_ = normalized + offset * t ;
        T * restrict acc2_ = acc2 + offset * t ;
        T * end = acc + width*height ;
        for(T * restrict acc_ = acc ; acc_ != end ;
            ++acc_, ++acc2_, ++data_, ++dzdy_, ++normalized_) {
          T L = kappa + alpha * (*acc_) ;
          T Lbeta = fast_pow(L, -beta) ;
          T temp1 = (*dzdy_) * Lbeta ;
          T temp2 = (*data_) * ab2 * temp1 / L ;
          *normalized_ = temp1 ;
          *acc2_ = temp2 ;
        }
      }
    }

    /*
     Integrate along feature channels in acc2, summing plane t-1 to
     plane t.
     */
    for (t = 1 ; t < (signed)depth ; ++t) {
      T * restrict acc2_ = acc2 + t * offset ;
      T const* restrict src_ = acc2_ - offset ;
      T const* end = acc2_ + offset ;
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
      T const* restrict acc22_ = acc2 + offset * q2 ;
      T const* restrict acc21_ = acc2 + offset * q1 ;
      T const* restrict data_  = data + offset * t ;
      T const* restrict end = data_  + width*height ;
      T * restrict normalized_ = normalized + offset * t ;
      if (q1 >= 0) {
        for( ; data_ != end ; ++data_, ++acc22_, ++acc21_, ++normalized_) {
          *normalized_ -= (*acc22_ - *acc21_) * (*data_) ;
        }
      } else {
        for( ; data_ != end ; ++data_, ++acc22_, ++normalized_) {
          *normalized_ -= (*acc22_) * (*data_) ;
        }
      }
    }
    data += width*height*depth ;
    normalized += width*height*depth ;
    dzdy += width*height*depth ;
  }
  free(acc) ;
  free(acc2) ;
#endif
}

template<> vl::Error
vl::impl::normalize_backward<vl::CPU, float>
(float* derData,
 float const* data,
 float const* derNormalized,
 size_t height, size_t width, size_t depth, size_t size,
 size_t normDepth,
 double kappa, double alpha, double beta)
{
  normalize_backward_cpu<float>(derData,data,derNormalized,
                                height,width,depth,size,
                                normDepth,kappa,alpha,beta) ;
  return vlSuccess ;
}
