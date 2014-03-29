//
//  normalize.cpp
//  matconv
//
//  Created by Andrea Vedaldi on 28/03/2014.
//  Copyright (c) 2014 Andrea Vedaldi. All rights reserved.
//

#include "normalize.h"

/* ---------------------------------------------------------------- */
/*                                                  normalize (CPU) */
/* ---------------------------------------------------------------- */

template<typename T>
void normalize_cpu(T* normalized,
                   T const* data,
                   size_t width,
                   size_t height,
                   size_t depth,
                   size_t normDetph,
                   T kappa, T alpha, T beta)
{
  size_t m = (normDepth-1)/2 ;
  for (int h = 0 ; h < height ; ++h) {
    for (int w = 0 ; w < width ; ++w) {
      int c ;
      T* x = data + w + h * width ;
      T* y = normalized + w + h * width ;
      T acc ;
      /* window [0, m-1] */
      for (c = 0 ; c < m ; ++c) {
        T a = *x++ ;
        acc += a*a ;
      }
      /* window [0, ..., t, ..., t+m] + t, 0<=t<m */
      for (c = 0 ; c < 2*m-1 ; ++c) {
        T a = *x++ ;
        acc += a*a ;
        *y++ = a * pow(kappa + alpha * acc, -beta) ;
      }
      /* window [t+m, ..., t, ..., t+m] + t, m<=t<depth-m */
      for ( ; c < depth-m ; ++c) {
        T a_ = *(x - 2*m - 1) ;
        T a = *x++ ;
        acc += a*a - a_*a_ ;
        *y++ = a * pow(kappa + alpha * acc, -beta) ;
      }
      /* window [t+m, ..., t, ..., m-1] + t, depth-m<=t<depth-1 */
      for ( ; c < depth ; ++c) {
        T a_ = *(x - 2*m - 1) ;
        x++ ;
        acc -= a_*a_ ;
        *y++ = a * pow(kappa + alpha * acc, -beta) ;
      }
    }
  }
}
