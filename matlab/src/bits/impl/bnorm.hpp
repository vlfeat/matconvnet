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

  template<vl::Device dev, typename type>
  struct bnorm
  {
    static vl::Error
    forward(Context& context,
            type* output,
            type* moments, // can be null and it will be allocated internally
            type const* data,
            type const* multipliers,
            type const* biases,
            int height, int width, int depth, int size,
            type epsilon) ;

    static vl::Error
    forward_given_moments(Context& context,
                          type* output,
                          type const* moments,
                          type const* data,
                          type const* multipliers,
                          type const* biases,
                          int height, int width, int depth, int size) ;

    static vl::Error
    backward(Context& context,
             type* derData,
             type* derMultipliers,
             type* derBiases,
             type* moments,
             type const* data,
             type const* multipliers,
             type const* biases,
             type const* derOutput,
             int height, int width, int depth, int size,
             type epsilon) ;

    static vl::Error
    backward_given_moments(Context& context,
                           type* derData,
                           type* derMultipliers,
                           type* derBiases,
                           type const* moments,
                           type const* data,
                           type const* multipliers,
                           type const* biases,
                           type const* derOutput,
                           int height, int width, int depth, int size,
                           type epsilon) ;
  } ;

} }
#endif /* __vl__bnorm__ */
