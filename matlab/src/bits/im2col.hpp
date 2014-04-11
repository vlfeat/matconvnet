/** @file im2col.hpp
 ** @brief Image to columns and back
 ** @author Andrea Vedaldi
 **/

#ifndef __matconv__im2col__
#define __matconv__im2col__

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
                const int height, const int width, const int kheight, const int kwidth,
                const int pad, const int stride,
                Dtype* data_col) ;

#if 0
template <typename Dtype>
void im2row_cpu(const Dtype* data_im,
                const int channels, const int height, const int width,
                const int kheight, const int kwidth,
                const int pad, const int stride,
                Dtype* data_col) ;
#endif

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
                const int height, const int width, const int kheight, const int kwidth,
                const int pad, const int stride,
                Dtype* data_im) ;


#ifdef ENABLE_GPU
template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int channels,
                const int height, const int width, const int ksize,
                const int pad, const int stride,
                Dtype* data_col) ;

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
                const int height, const int width, const int ksize,
                const int pad, const int stride,
                Dtype* data_im) ;
#endif

#endif /* defined(__matconv__im2col__) */
