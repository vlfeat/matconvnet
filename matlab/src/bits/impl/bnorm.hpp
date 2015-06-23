// @file bnorm.hpp
// @brief Batch Normalization block implementation
// @author Sebastien Ehrhardt

/*
Copyright (C) 2015 Sebastien Ehrhardt.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__bnorm__
#define __vl__bnorm__

#include "../data.hpp"
#include <cstddef>

namespace vl { namespace impl {

  template<vl::Device dev, typename type> vl::Error
  bnorm_forward(Context& context,
                type* output,
                type const* data,
                type const* multipliers,
                type const* biases,
                int height, int width, int depth, int size, type epsilon) ;


  template<vl::Device dev, typename type> vl::Error
  bnorm_backward(Context& context,
                 type* derData,
                 type* derMultipliers,
                 type* derBiaises,
                 type const* data,
                 type const* multipliers,
                 type const* biases,
                 type const* derOutput,
                 int height, int width, int depth, int size,
                 float epsilon) ;


  /* Specializations: CPU, float */

  template<> vl::Error
  bnorm_forward<vl::CPU, float>(Context& context,
                                float* output,
                                float const* data,
                                float const* multipliers,
                                float const* biases,
                                int height, int width, int depth, int size, float epsilon) ;

  template<> vl::Error
  bnorm_backward<vl::CPU, float>(Context& context,
                                 float* derData,
                                 float* derMultipliers,
                                 float* derBiaises,
                                 float const* data,
                                 float const* multipliers,
                                 float const* biases,
                                 float const* derOutput,
                                 int height, int width, int depth, int size,
                                 float epsilon)  ;

  /* Specializations: GPU, float */

#if ENABLE_GPU
  template<> vl::Error
  bnorm_forward<vl::GPU, float>(Context& context,
                                float* output,
                                float const* data,
                                float const* multipliers,
                                float const* biases,
                                int height, int width, int depth, int size, float epsilon) ;

  template<> vl::Error
  bnorm_backward<vl::GPU, float>(Context& context,
                                 float* derData,
                                 float* derMultipliers,
                                 float* derBiaises,
                                 float const* data,
                                 float const* multipliers,
                                 float const* biases,
                                 float const* derOutput,
                                 int height, int width, int depth, int size,
                                 float epsilon) ;
#endif
  
} }
#endif /* __vl__bnorm__ */
