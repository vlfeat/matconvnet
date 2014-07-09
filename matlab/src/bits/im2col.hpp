/** @file im2col.hpp
 ** @brief Image to columns and back
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2014 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __matconv__im2col__
#define __matconv__im2col__

#include <ctype.h>

template <typename T>
void im2col_cpu(T* stacked,
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

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
                const int height, const int width, const int kheight, const int kwidth,
                const int pad, const int stride,
                Dtype* data_im) ;


#ifdef ENABLE_GPU
template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int channels,
                const int height, const int width,
                const int kheight, const int kwidth,
                const int pad, const int stride,
                Dtype* data_col) ;

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
                const int height, const int width,
                const int kheight, const int kwidth,
                const int pad, const int stride,
                Dtype* data_im) ;
#endif

#endif /* defined(__matconv__im2col__) */
