//
//  im2col.h
//  matconv
//
//  Created by Andrea Vedaldi on 09/02/2014.
//  Copyright (c) 2014 Andrea Vedaldi. All rights reserved.
//

#ifndef __matconv__im2col__
#define __matconv__im2col__

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
                const int height, const int width, const int ksize, const int stride,
                Dtype* data_col) ;

template <typename Dtype>
void im2row_cpu(const Dtype* data_im,
                const int channels, const int height, const int width,
                const int ksize, const int stride,
                Dtype* data_col) ;

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
                const int height, const int width, const int ksize, const int stride,
                Dtype* data_im) ;

template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int channels,
                const int height, const int width, const int ksize, const int stride,
                Dtype* data_col) ;

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
                const int height, const int width, const int ksize, const int stride,
                Dtype* data_im) ;

#endif /* defined(__matconv__im2col__) */
