/** @file pooling.hpp
 ** @brief Max pooling filters
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2014 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __matconv__pooling__
#define __matconv__pooling__

#include <cstddef>

template<typename T>
void maxPooling_cpu(T* pooled,
                    T const* data,
                    size_t width,
                    size_t height,
                    size_t depth,
                    size_t windowWidth,
                    size_t windowHeight,
                    size_t strideX,
                    size_t strideY,
                    size_t padLeft,
                    size_t padRight,
                    size_t padTop,
                    size_t padBottom) ;

template<typename T>
void maxPoolingBackward_cpu(T* dzdx,
                            T const* data,
                            T const* dzdy,
                            size_t width,
                            size_t height,
                            size_t depth,
                            size_t windowWidth,
                            size_t windowHeight,
                            size_t strideX,
                            size_t strideY,
                            size_t padLeft,
                            size_t padRight,
                            size_t padTop,
                            size_t padBottom) ;

#ifdef ENABLE_GPU
template<typename T>
void maxPooling_gpu(T* pooled,
                    T const* data,
                    size_t width,
                    size_t height,
                    size_t depth,
                    size_t poolSize,
                    size_t stride,
                    size_t pad) ;

template<typename T>
void maxPoolingBackward_gpu(T* dzdx,
                            T const* data,
                            T const* dzdy,
                            size_t width,
                            size_t height,
                            size_t depth,
                            size_t poolSize,
                            size_t stride,
                            size_t pad) ;
#endif

#endif /* defined(__matconv__pooling__) */
