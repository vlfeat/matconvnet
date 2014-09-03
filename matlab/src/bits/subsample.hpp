/** @file subsample.hpp
 ** @brief Subsampling operator
 ** @author Andrea Vedaldi
**/

/*
 Copyright (C) 2014 Andrea Vedaldi and Karel Lenc.
 All rights reserved.

 This file is part of the VLFeat library and is made available under
 the terms of the BSD license (see the COPYING file).
 */

#ifndef VL_NNSUBSAMPLE_H
#define VL_NNSUBSAMPLE_H

#include <cstddef>

template<typename T>
void subsample_cpu(T* subsampled,
                   T const* data,
                   size_t width,
                   size_t height,
                   size_t depth,
                   size_t strideX,
                   size_t strideY,
                   size_t padLeft,
                   size_t padRight,
                   size_t padTop,
                   size_t padBottom) ;

template<typename T>
void subsampleBackward_cpu(T* dzdx,
                           T const* dzdy,
                           size_t width,
                           size_t height,
                           size_t depth,
                           size_t strideX,
                           size_t strideY,
                           size_t padLeft,
                           size_t padRight,
                           size_t padTop,
                           size_t padBottom) ;

#ifdef ENABLE_GPU
template<typename T>
void subsample_gpu(T* subsampled,
                   T const* data,
                   size_t width,
                   size_t height,
                   size_t depth,
                   size_t strideX,
                   size_t strideY,
                   size_t padLeft,
                   size_t padRight,
                   size_t padTop,
                   size_t padBottom) ;

template<typename T>
void subsampleBackward_gpu(T* dzdx,
                           T const* dzdy,
                           size_t width,
                           size_t height,
                           size_t depth,
                           size_t strideX,
                           size_t strideY,
                           size_t padLeft,
                           size_t padRight,
                           size_t padTop,
                           size_t padBottom) ;
#endif

#endif /* defined(VL_NNSUBSAMPLE_H) */
