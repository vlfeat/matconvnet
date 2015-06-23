// @file   bnorm_cpu.cpp
// @brief  Batch normalization implementation (CPU)
// @author Sebastien Ehrhardt

/*
Copyright (C) 2015 Sebastien Ehrhardt.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "bnorm.hpp"
#include "../data.hpp"
#include <math.h>
#include <memory.h>
#include <cstdlib>
#include <algorithm>
#include <limits>

/* ---------------------------------------------------------------- */
/*                                              bnorm forward       */
/* ---------------------------------------------------------------- */

template<typename T> inline void
computeMoments(T const * data,
               T  * mean,
               T  * sigma,
               T  epsilon,
               unsigned int WH,
               unsigned int depth,
               unsigned int num)
{
  unsigned int mass = WH*num;
  for(int channel = 0; channel < depth; ++channel) {
    for(int element = 0; element < num; ++element) {
      for(int wh = 0; wh < WH; ++wh){
        mean[channel]+=data[wh+channel*WH+WH*depth*element];
        sigma[channel]+= data[wh+channel*WH+WH*depth*element]*data[wh+channel*WH+WH*depth*element];
      }
    }
  }

  for(int i = 0; i < depth; ++i) {
    mean[i] /= mass;
    sigma[i] = sqrt(sigma[i]/mass - mean[i]*mean[i] + epsilon);
  }
}

template<typename T> inline void
batchNormalizeForward(T * output,
                      T const* data,
                      T * mean,
                      T * sigma,
                      T * filters,
                      T * biaises,
                      unsigned int  WH,
                      size_t depth,
                      size_t num)
{
  T f, b;
  for(int channel =0; channel < depth; ++channel) {
    f = filters[channel]/sigma[channel];
    b = biaises[channel];
    for(int element = 0; element < num; ++element) {
      for(int wh = 0; wh < WH; ++wh){
        output[wh+channel*WH+WH*depth*element] = f*(data[wh+channel*WH+WH*depth*element]-mean[channel])+b;
      }
    }
  }
}

template<typename T> static inline void
bnorm_forward_cpu(T* output,
                  T const* data,
                  T * filters,
                  T * biaises,
                  size_t width,
                  size_t height,
                  size_t depth,
                  size_t num,
                  T epsilon)
{
  // First Compute Mu and Sigma together
  T * mean, * sigma;
  mean = (T*)calloc(sizeof(T),depth);
  sigma = (T*)calloc(sizeof(T),depth);
  unsigned int WH =(unsigned int)width*height;
  computeMoments<T>(data, mean, sigma, epsilon, WH, depth, num);

  // Batch Normalize
  batchNormalizeForward<T>(output, data, mean, sigma, filters, biaises, WH, depth, num);

  // Delete intermediate variable
  free(mean);
  free(sigma);
}

template<> vl::Error
vl::impl::bnorm_forward<vl::CPU, float>(Context& context, float* output,
                                        float const* data,
                                        float* filters,
                                        float* biaises,
                                        size_t height, size_t width, size_t depth, size_t size, float epsilon)
{
  bnorm_forward_cpu<float>(output, data,
                           filters, biaises,
                           height, width,
                           depth, size, epsilon) ;

  return vlSuccess;
}

/* ---------------------------------------------------------------- */
/*                                              bnorm backward      */
/* ---------------------------------------------------------------- */

template<typename T> inline void
computeDer(T * derOutput,
           T const * data,
           T * derFilters,
           T * derBiaises,
           T * muz,
           T * mean,
           T * sigma,
           T epsilon,
           unsigned int WH,
           unsigned int depth,
           unsigned int num)
{
  unsigned int mass = WH*num;
  memset(derFilters, 0, sizeof(T) * depth) ;
  memset(derBiaises, 0, sizeof(T) * depth) ;
  for(int channel = 0; channel < depth; ++channel){
    for(int element = 0; element < num; ++element ){
      for(int wh = 0; wh < WH; ++wh){
        mean[channel] += data[wh+channel*WH+WH*depth*element];
        sigma[channel] += data[wh+channel*WH+WH*depth*element]*data[wh+channel*WH+WH*depth*element];
        derFilters[channel] += derOutput[wh+channel*WH+WH*depth*element]*data[wh+channel*WH+WH*depth*element];
        derBiaises[channel] += derOutput[wh+channel*WH+WH*depth*element];
      }
    }
  }

  for(int i = 0; i < depth; ++i){
    mean[i] /= mass;
    sigma[i] = sqrt(sigma[i]/mass -mean[i]*mean[i]+epsilon);
    derFilters[i] = (derFilters[i]-derBiaises[i]*mean[i])/sigma[i];
    muz[i] = derBiaises[i]/mass;
  }
}


template<typename T> inline void
batchNormalizeBackward(T * derData,
                       T * derFilters,
                       T * muz,
                       T const * data,
                       T const * filters,
                       T * mean,
                       T * sigma,
                       unsigned int  WH,
                       size_t depth,
                       size_t num,
                       T * derOutput)
{
  unsigned int mass = WH*num;
  T G1, G2;
  for(int channel =0; channel < depth; ++channel ) {
    G1 = filters[channel]/sigma[channel];
    G2 = derFilters[channel]/(mass*sigma[channel]);
    for(int element = 0; element < num; ++element){
      for(int wh = 0; wh < WH; ++wh){
        derData[wh+channel*WH+WH*depth*element] = G1*((derOutput[wh+channel*WH+WH*depth*element]-muz[channel])-G2*(data[wh+channel*WH+WH*depth*element]-mean[channel]));
      }
    }
  }
}

template<typename T> static inline void
bnorm_backward_cpu(T* derData,
                   T * derFilters,
                   T * derBiaises,
                   T const* data,
                   T const * filters,
                   T const * biaises,
                   size_t width,
                   size_t height,
                   size_t depth,
                   size_t num,
                   T * derOutput,
                   T epsilon)
{
  // First Compute Mu and Sigma together
  T * mean, * sigma, * muz;
  mean = (T*)calloc(sizeof(T),depth);
  sigma = (T*)calloc(sizeof(T),depth);
  muz = (T*)calloc(sizeof(T),depth);
  unsigned int WH =(unsigned int)width*height;

  // Compute derFilters and derBiaises and Moments at the same time
  computeDer<T>(derOutput, data, derFilters, derBiaises,
                muz, mean, sigma, epsilon, WH, depth, num);

  //Batch Normalize
  batchNormalizeBackward<T>(derData, derFilters, muz,
                            data, filters, mean, sigma,
                            WH, depth, num, derOutput);

  // Delete intermediate variable
  free(mean);
  free(sigma);
}

template<> vl::Error
vl::impl::bnorm_backward<vl::CPU, float>(Context& context, float* derData,
                                         float* derFilters,
                                         float* derBiaises,
                                         float const* data,
                                         float const* filters,
                                         float const* biaises,
                                         size_t height, size_t width, size_t depth, size_t size,
                                         float* derOutput,
                                         float epsilon)
{
  bnorm_backward_cpu<float>(derData, derFilters,derBiaises,
                            data, filters, biaises, width, height,
                            depth, size, derOutput, epsilon);

  return vlSuccess;
}

