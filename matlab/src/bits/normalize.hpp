/** @file normalize.hpp
 ** @brief Normalization
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2014 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __matconv__normalize__
#define __matconv__normalize__

#include <cstddef>

template<typename T>
void normalize_cpu(T* pooled,
                   T const* data,
                   size_t width,
                   size_t height,
                   size_t depth,
                   size_t num,
                   size_t normDetph,
                   T kappa, T alpha, T beta) ;

template<typename T>
void normalizeBackward_cpu(T* dzdx,
                           T const* data,
                           T const* dzdy,
                           size_t width,
                           size_t height,
                           size_t depth,
                           size_t num,
                           size_t normDetph,
                           T kappa, T alpha, T beta) ;

#ifdef ENABLE_GPU
template<typename T>
void normalize_gpu(T* pooled,
                   T const* data,
                   size_t width,
                   size_t height,
                   size_t depth,
                   size_t num,
                   size_t normDetph,
                   T kappa, T alpha, T beta) ;

template<typename T>
void normalizeBackward_gpu(T* dzdx,
                           T const* data,
                           T const* dzdy,
                           size_t width,
                           size_t height,
                           size_t depth,
                           size_t num,
                           size_t normDetph,
                           T kappa, T alpha, T beta) ;
#endif

#endif /* defined(__matconv__normalize__) */
