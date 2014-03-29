//
//  normalize.cpp
//  matconv
//
//  Created by Andrea Vedaldi on 28/03/2014.
//  Copyright (c) 2014 Andrea Vedaldi. All rights reserved.
//

#include "normalize.hpp"

/* ---------------------------------------------------------------- */
/*                                                  normalize (CPU) */
/* ---------------------------------------------------------------- */

template<typename T>
void normalize_cpu(T* normalized,
                   T const* data,
                   size_t width,
                   size_t height,
                   size_t depth,
                   size_t normDepth,
                   T kappa, T alpha, T beta)
{
  int m = ((signed)normDepth-1)/2 ;
  for (int h = 0 ; h < height ; ++h) {
    for (int w = 0 ; w < width ; ++w) {
      int c ;
      T const* x = data + w + h * width ;
      T* y = normalized + w + h * width ;
      T acc = 0 ;
      /* window [0, m-1] */
      for (c = 0 ; c < m ; ++c) {
        T ap = *x++ ;
        acc += ap*ap ;
      }
      /* window [0, ..., t, ..., t+m] + t, 0<=t<m */
      for ( ; c < 2*m-1 ; ++c) {
        T a = *(x - m) ;
        T ap = *x++ ;
        acc += ap*ap ;
        *y++ = a * pow(kappa + alpha * acc, -beta) ;
      }
      /* window [t+m, ..., t, ..., t+m] + t, m<=t<depth-m */
      for ( ; c < depth-m ; ++c) {
        T am = *(x - 2*m - 1) ;
        T a  = *(x - m) ;
        T ap = *x++ ;
        acc += ap*ap - am*am ;
        *y++ = a * pow(kappa + alpha * acc, -beta) ;
      }
      /* window [t+m, ..., t, ..., m-1] + t, depth-m<=t<depth-1 */
      for ( ; c < depth ; ++c) {
        T am = *(x - 2*m - 1) ;
        T a  = *(x - m) ;
        x++ ;
        acc -= am*am ;
        *y++ = a * pow(kappa + alpha * acc, -beta) ;
      }
    }
  }
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
