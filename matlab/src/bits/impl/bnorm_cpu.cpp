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
/*                                                    bnorm forward */
/* ---------------------------------------------------------------- */

/* 
 WH is the product of the data width and height
 mean[] and sigma[] must have a number of dimensions equal to depth
 */
template<typename T> inline void
computeMoments(T const * data,
               T * mean,
               T * sigma,
               T epsilon,
               int WH,
               int depth,
               int num)
{
  int mass = WH * num ;
  for(int channel = 0; channel < depth; ++channel) {
    for(int element = 0; element < num; ++element) {
      for(int wh = 0; wh < WH; ++wh){
        T x = data[wh + channel*WH + element*(depth*WH)] ;
        mean[channel] += x;
        sigma[channel] += x * x;
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
                      T const* mean,
                      T const* sigma,
                      T const* multipliers,
                      T const* biases,
                      int WH,
                      int depth,
                      int num)
{
  T f;
  T b;
  for(int channel = 0; channel < depth; ++channel) {
    f = multipliers[channel] / sigma[channel];
    b = biases[channel];
    for(int element = 0; element < num; ++element) {
      for(int wh = 0; wh < WH; ++wh){
        int offset = wh + channel*WH + element * (depth*WH) ;
        output[offset] = f * (data[offset]-mean[channel]) + b ;
      }
    }
  }
}

template<typename T> static inline void
bnorm_forward_cpu(T* output,
                  T const* data,
                  T const* multipliers,
                  T const* biases,
                  int width,
                  int height,
                  int depth,
                  int num,
                  T epsilon)
{
  // First Compute Mu and Sigma together
  T * mean, * sigma;
  mean = (T*)calloc(sizeof(T),depth);
  sigma = (T*)calloc(sizeof(T),depth);
  int WH = width*height;
  computeMoments<T>(data, mean, sigma, epsilon, WH, depth, num);

  // Batch Normalize
  batchNormalizeForward<T>(output, data, mean, sigma, multipliers, biases, WH, depth, num);

  // Delete intermediate variable
  free(mean);
  free(sigma);
}

template<> vl::Error
vl::impl::bnorm_forward<vl::CPU, float>(Context& context,
                                        float* output,
                                        float const* data,
                                        float const* multipliers,
                                        float const* biases,
                                        int height, int width,
                                        int depth, int size,
                                        float epsilon)
{
  bnorm_forward_cpu<float>(output, data,
                           multipliers, biases,
                           height, width,
                           depth, size, epsilon) ;

  return vlSuccess;
}

/* ---------------------------------------------------------------- */
/*                                              bnorm backward      */
/* ---------------------------------------------------------------- */

template<typename T> inline void
computeDer(T * derMultipliers,
           T * derBiases,
           T * muz,
           T * mean,
           T * sigma,
           T const * data,
           T const * derOutput,
           int WH,
           int depth,
           int num,
           T epsilon)
{
  int mass = WH*num;
  memset(derMultipliers, 0, sizeof(T) * depth) ;
  memset(derBiases, 0, sizeof(T) * depth) ;
  for(int channel = 0; channel < depth; ++channel){
    for(int element = 0; element < num; ++element ){
      for(int wh = 0; wh < WH; ++wh){
        int offset = wh + channel*WH + element * (WH*depth) ;
        mean[channel] += data[offset] ;
        sigma[channel] += data[offset] * data[offset];
        derMultipliers[channel] += derOutput[offset] * data[offset];
        derBiases[channel] += derOutput[offset];
      }
    }
  }

  for(int i = 0; i < depth; ++i){
    mean[i] /= mass;
    sigma[i] = sqrt(sigma[i]/mass - mean[i]*mean[i]+epsilon);
    derMultipliers[i] = (derMultipliers[i] - derBiases[i]*mean[i])/sigma[i];
    muz[i] = derBiases[i]/mass;
  }
}


template<typename T> inline void
batchNormalizeBackward(T * derData,
                       T * derMultipliers,
                       T * muz,
                       T const * data,
                       T const * multipliers,
                       T const * mean,
                       T const * sigma,
                       T const * derOutput,
                       int WH,
                       int depth,
                       int num)
{
  int mass = WH*num;
  T G1, G2;
  for(int channel =0; channel < depth; ++channel ) {
    G1 = multipliers[channel]/sigma[channel];
    G2 = derMultipliers[channel]/(mass*sigma[channel]);
    for(int element = 0; element < num; ++element){
      for(int wh = 0; wh < WH; ++wh){
        int offset = wh + channel*WH + element * (WH*depth) ;
        derData[offset] = G1 *
        ((derOutput[offset] - muz[channel]) - G2 * (data[offset]-mean[channel]));
      }
    }
  }
}

template<typename T> static inline void
bnorm_backward_cpu(T * derData,
                   T * derMultipliers,
                   T * derBiases,
                   T const * data,
                   T const * multipliers,
                   T const * biases,
                   T const * derOutput,
                   int width,
                   int height,
                   int depth,
                   int num,
                   T epsilon)
{
  // First Compute Mu and Sigma together
  T * mean, * sigma, * muz;
  mean = (T*)calloc(sizeof(T),depth);
  sigma = (T*)calloc(sizeof(T),depth);
  muz = (T*)calloc(sizeof(T),depth);
  int WH = width * height;

  // Compute derMultipliers and derBiases and Moments at the same time
  computeDer<T>(derMultipliers, derBiases,
                muz, mean, sigma,
                data, derOutput,
                WH, depth, num,
                epsilon);

  //Batch Normalize
  batchNormalizeBackward<T>(derData, derMultipliers, muz,
                            data, multipliers, mean, sigma, derOutput,
                            WH, depth, num);

  // Delete intermediate variable
  free(mean);
  free(sigma);
}

template<> vl::Error
vl::impl::bnorm_backward<vl::CPU, float>(Context& context,
                                         float* derData,
                                         float* derMultipliers,
                                         float* derBiases,
                                         float const* data,
                                         float const* multipliers,
                                         float const* biases,
                                         float const* derOutput,
                                         int height, int width, int depth, int size,
                                         float epsilon)
{
  bnorm_backward_cpu<float>(derData, derMultipliers,derBiases,
                            data, multipliers, biases, derOutput,
                            width, height, depth, size, epsilon);
  return vlSuccess;
}

