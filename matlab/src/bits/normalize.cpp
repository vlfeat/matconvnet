/** @file normalize.cpp
 ** @brief Normalization block
 ** @author Andrea Vedaldi
 **/

#include "normalize.hpp"
#include <algorithm>
#include <cmath>
#include <blas.h>
#include <string.h>

/* ---------------------------------------------------------------- */
/*                                                  normalize (CPU) */
/* ---------------------------------------------------------------- */

#pragma GCC optimize ("fast-math")
/* #define VL_NNNORMALIZE_FAST */
#define max(a,b) (((a)>=(b))?a:b)
#define xat(t) x[(t) * offset]
#define yat(t) y[(t) * offset]
#define zat(t) z[(t) * offset]

inline float pow_forward(float a, float b) { return __builtin_powf(a,b) ; }
inline double pow_forward(double a, double b) { return __builtin_pow(a,b) ; }

#ifndef VL_NNNORMALIZE_FAST
inline double fast_pow(double a, double b) { return pow(a,b) ; }
inline float fast_pow(float a, float b) { return powf(a,b) ; }
#else
inline double fast_pow(double a, double b)
{
  union {
    double d;
    int x[2];
  } u = { a };
  const int offset = 1023 << 20 ;
  u.x[1] = (int)(b * (u.x[1] - offset) + offset) ;
  u.x[0] = 0;
  return u.d;
}
inline float fast_pow(float a, float b)
{
  union {
    float d;
    int x ;
  } u = { a };
  const int offset = 127 << 23 ;
  u.x = (int)(b * (u.x - offset) + offset) ;
  return u.d;
}
#endif

inline void sbmv_forward(char * uplo,
                         ptrdiff_t *n,
                         ptrdiff_t *k,
                         float * alpha,
                         float * a,
                         ptrdiff_t *lda,
                         float * x,
                         ptrdiff_t *incx,
                         float * beta,
                         float * y,
                         ptrdiff_t *incy)
{
  ssbmv(uplo,n,k,alpha,a,lda,x,incx,beta,y,incy) ;
}

inline void sbmv_forward(char * uplo,
                         ptrdiff_t *n,
                         ptrdiff_t *k,
                         double * alpha,
                         double * a,
                         ptrdiff_t *lda,
                         double * x,
                         ptrdiff_t *incx,
                         double * beta,
                         double * y,
                         ptrdiff_t *incy)
{
  dsbmv(uplo,n,k,alpha,a,lda,x,incx,beta,y,incy) ;
}

template<typename T>
void normalize_cpu(T* normalized,
                   T const* data,
                   size_t width,
                   size_t height,
                   size_t depth,
                   size_t normDepth,
                   T kappa, T alpha, T beta)
{
  int t ;
  int m1 = ((signed)normDepth-1)/2 ;
  int m2 = normDepth - m1 - 1 ;
  int offset = width*height ;
#ifndef VL_NNNORMALIZE_FAST
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
          yat(t) = xat(t) * pow(kappa + alpha * acc, -beta) ;
        }
      }
    }
  }
#elif 0
  T * acc = (T*) calloc(sizeof(T), width*height) ;
  for (t = -m2 ; t < (signed)depth ; ++t) {
    int tm = t - m1 - 1 ;
    int tp = t + m2 ;
    char L = 'L' ;
    ptrdiff_t bandSize = 0 ;
    ptrdiff_t n = width*height ;
    ptrdiff_t p = 1 ;
    T one = 1 ;
    T minusOne = -1 ;

    if (tp < depth) {
      T const* xap = data + offset * (t+m2) ;
      sbmv_forward(&L, &n, &bandSize,
                    &one, /* alpha*/
                   (T*) xap, &p, /* A, lda */
                   (T*) xap, &p, /* x, incx */
                   &one, /* beta */
                   acc, &p) ; /* y, incy */
    }

    if (0 <= tm) {
      T const* xam = data + offset * (t-m1-1) ;
      sbmv_forward(&L, &n, &bandSize,
                   &minusOne, /* alpha*/
                   (T*) xam, &p, /* A, lda */
                   (T*) xam, &p, /* x, incx */
                   &one, /* one */
                   acc, &p) ; /* y, incy */
    }

    if (0 <= t && t < depth) {
      T const* xx = data + offset * t ;
      T* xy = normalized + offset * t ;
      T* end = acc + n ;
      for(T *xacc = acc ; xacc != end ; ++xacc, ++xx, ++xy) {
        //        (*xy) = (*xx) * pow_forward(kappa + alpha * (*xacc), -beta) ;
        (*xy) = (*xx) * fast_pow(kappa + alpha * (*xacc), -beta) ;
      }
    }

  }
  free(acc) ;
#else
  T * acc = (T*) calloc(sizeof(T), width*height) ;
  for (t = -m2 ; t < (signed)depth ; ++t) {
    int tm = t - m1 - 1 ;
    int tp = t + m2 ;
    T const* xam = data + offset * (t-m1-1) ;
    T const* xap = data + offset * (t+m2) ;
    T *end = acc + width*height ;
    if (0 <= tm & tp < depth) {
      for(T *xacc = acc ; xacc != end ; ++xacc, ++xam, ++xap) {
        T am = *xam ;
        T ap = *xap ;
        *xacc += ap*ap - am*am ;
      }
    } else if (0 > tm & tp < depth) {
      for(T *xacc = acc ; xacc != end ; ++xacc, ++xap) {
        T ap = *xap ;
        *xacc += ap*ap ;
      }
    } else if (0 <= tm & tp >= depth) {
      for(T *xacc = acc ; xacc != end ; ++xacc, ++xam) {
        T am = *xam ;
        *xacc -= am*am ;
      }
    }
    if (0 <= t && t < depth) {
      T const* xx = data + offset * t ;
      T* xy = normalized + offset * t ;
      for(T *xacc = acc ; xacc != end ; ++xacc, ++xx, ++xy) {
        //        (*xy) = (*xx) * pow_forward(kappa + alpha * (*xacc), -beta) ;
        (*xy) = (*xx) * fast_pow(kappa + alpha * (*xacc), -beta) ;
      }
    }
  }
  free(acc) ;
#endif
}

template
void normalize_cpu<float>(float* normalized,
                          float const* data,
                          size_t width,
                          size_t height,
                          size_t depth,
                          size_t normDetph,
                          float kappa, float alpha, float beta) ;

template
void normalize_cpu<double>(double* normalized,
                           double const* data,
                           size_t width,
                           size_t height,
                           size_t depth,
                           size_t normDetph,
                           double kappa, double alpha, double beta) ;


/* ---------------------------------------------------------------- */
/*                                                  normalize (CPU) */
/* ---------------------------------------------------------------- */

template<typename T>
void normalizeBackward_cpu(T* normalized,
                           T const* data,
                           T const* dzdy,
                           size_t width,
                           size_t height,
                           size_t depth,
                           size_t normDepth,
                           T kappa, T alpha, T beta)
{
  int m1 = ((signed)normDepth-1)/2 ;
  int m2 = normDepth - m1 - 1 ;
  int offset = width*height ;
  T ab2 = 2*alpha*beta ;
  int t, q ;

#ifndef VL_NNNORMALIZE_FAST
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
        T Lbeta = pow(L, -beta) ;
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
#elif 0
  T * acc = (T*) calloc(sizeof(T), width*height) ;
  memset(normalized, 0, sizeof(T) * height*width) ;
  for (t = -m2 ; t < (signed)depth ; ++t) {
    int tm = t - m1 - 1 ;
    int tp = t + m2 ;
    int q1 = t - m1 ;
    int q2 = t + m2 ;
    T const* xam = data + offset * (t-m1-1) ;
    T const* xap = data + offset * (t+m2) ;
    T *end = acc + width*height ;

    if (0 <= tm & tp < depth) {
      for(T *xacc = acc ; xacc != end ; ++xacc, ++xam, ++xap) {
        T am = *xam ;
        T ap = *xap ;
        *xacc += ap*ap - am*am ;
      }
    } else if (0 > tm & tp < depth) {
      q1 = 0 ;
      for(T *xacc = acc ; xacc != end ; ++xacc, ++xap) {
        T ap = *xap ;
        *xacc += ap*ap ;
      }
    } else if (0 <= tm & tp >= depth) {
      q2 = depth - 1 ;
      for(T *xacc = acc ; xacc != end ; ++xacc, ++xam) {
        T am = *xam ;
        *xacc -= am*am ;
      }
    } else {
      q1 = 0 ;
      q2 = depth - 1 ;
    }
    if (0 <= t && t < depth) {
      T const* dataPt = data + offset * t ;
      T* normalizedPt = normalized + offset * t ;
      T const * xz = dzdy + offset * t ;
      for(T *xacc = acc ; xacc != end ; ++xacc, ++xx, ++xy, ++xz) {
        T L = kappa + alpha * (*xacc) ;
        T Lbeta = fast_pow(L, -beta) ;
        T Lbeta1 = Lbeta / L ;
        T z = *xz ;
        T coeff = z * ab2 * Lbeta1 * (*xx) ;
        *xy += z * Lbeta ;
        for (q = q1-t ; q <= q2-t ; ++ q) {
          xy[q * offset] -= coeff * xx[q* offset]  ;
        }
      }
    }
  }
  free(acc) ;
#else
  T * acc = (T*) calloc(sizeof(T), width*height) ;
  T * acc2 = (T*) malloc(sizeof(T) * width*height*depth) ;
  for (t = -m2 ; t < (signed)depth ; ++t) {
    int tm = t - m1 - 1 ;
    int tp = t + m2 ;
    T const* datam_ = data + offset * tm ;
    T const* datap_ = data + offset * tp ;
    T *end = acc + width*height ;

    if (0 <= tm & tp < depth) {
      for(T *acc_ = acc ; acc_ != end ; ++acc_, ++datap_, ++datam_) {
        T am = *datam_ ;
        T ap = *datap_ ;
        *acc_ += ap*ap - am*am ;
      }
    } else if (0 > tm & tp < depth) {
      for(T *acc_ = acc ; acc_ != end ; ++acc_, ++datap_) {
        T ap = *datap_ ;
        *acc_ += ap*ap ;
      }
    } else if (0 <= tm & tp >= depth) {
      for(T *acc_ = acc ; acc_ != end ; ++acc_, ++datam_) {
        T am = *datam_ ;
        *acc_ -= am*am ;
      }
    }
    if (0 <= t && t < depth) {
      T const* __restrict data_ = data + offset * t ;
      T const* __restrict dzdy_ = dzdy + offset * t ;
      T * __restrict normalized_ = normalized + offset * t ;
      T * __restrict acc2_ = acc2 + offset * t ;
      for(T * __restrict acc_ = acc ; acc_ != end ;
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
  for (t = 1 ; t < (signed)depth ; ++t) {
    T * __restrict acc2_ = acc2 + t * offset ;
    T const * __restrict src_ = acc2_ - offset ;
    T const * __restrict end = acc2_ + offset ;
    for( ; acc2_ != end ; ++acc2_, ++src_) {
      *acc2_ += *src_ ;
    }
  }
  for (t = 0 ; t < (signed)depth ; ++t) {
    int q1 = t - m2 - 1 ;
    int q2 = ((t + m1) <= (depth - 1)) ? t + m1 : depth - 1 ;
    T const* __restrict data_ = data + offset * t ;
    T const* __restrict end = data_  + width*height ;
    T const* __restrict acc22_ = acc2 + offset * q2 ;
    T const* __restrict acc21_ = acc2 + offset * q1 ;
    T * __restrict normalized_ = normalized + offset * t ;
#if 1
    if (q1 >= 0) {
      for( ; data_ != end ; ++data_, ++acc22_, ++acc21_, ++normalized_) {
        *normalized_ -= (*acc22_ - *acc21_) * (*data_) ;
      }
    } else {
      for( ; data_ != end ; ++data_, ++acc22_, ++normalized_) {
        *normalized_ -= (*acc22_) * (*data_) ;
      }
    }
#endif
  }
  free(acc) ;
  free(acc2) ;
#endif
}

template
void normalizeBackward_cpu<float>(float* normalized,
                                  float const* data,
                                  float const* dzdy,
                                  size_t width,
                                  size_t height,
                                  size_t depth,
                                  size_t normDetph,
                                  float kappa, float alpha, float beta) ;

template
void normalizeBackward_cpu<double>(double* normalized,
                                   double const* data,
                                   double const* dzdy,
                                   size_t width,
                                   size_t height,
                                   size_t depth,
                                   size_t normDetph,
                                   double kappa, double alpha, double beta) ;

