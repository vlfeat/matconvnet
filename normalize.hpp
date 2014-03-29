//
//  normalize.h
//  matconv
//
//  Created by Andrea Vedaldi on 28/03/2014.
//  Copyright (c) 2014 Andrea Vedaldi. All rights reserved.
//

#ifndef __matconv__normalize__
#define __matconv__normalize__

template<typename T>
void normalize_cpu(T* pooled,
                   T const data,
                   size_t width,
                   size_t height,
                   size_t depth,
                   size_t normDetph,
                   T kappa, T alpha, T beta) ;

template<typename T>
void normalizeBackward_cpu(T* dzdx,
                           T const* data,
                           T const* dzdy,
                           size_t width,
                           size_t height,
                           size_t depth,
                           size_t normDetph,
                           T kappa, T alpha, T beta) ;

template<typename T>
void normalize_gpu(T* pooled,
                   T const data,
                   size_t width,
                   size_t height,
                   size_t depth,
                   size_t normDetph,
                   T kappa, T alpha, T beta) ;

template<typename T>
void normalizeBackward_gpu(T* dzdx,
                           T const* data,
                           T const* dzdy,
                           size_t width,
                           size_t height,
                           size_t depth,
                           size_t normDetph,
                           T kappa, T alpha, T beta) ;

#endif /* defined(__matconv__normalize__) */
