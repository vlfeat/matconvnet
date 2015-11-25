// @file normalize.hpp
// @brief Normalize block implementation
// @author Andrea Vedaldi

/*
Copyright (C) 2014-15 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__normalize__
#define __vl__normalize__

#include "../data.hpp"
#include <cstddef>

namespace vl { namespace impl {

  template<vl::Device dev, typename type> vl::Error
  normalize_forward(type* normalized,
                    type const* data,
                    size_t height, size_t width, size_t depth, size_t size,
                    size_t normDetph,
                    double kappa, double alpha, double beta) ;

  template<vl::Device dev, typename type> vl::Error
  normalize_backward(type* derData,
                     type const* data,
                     type const* derNormalized,
                     size_t height, size_t width, size_t depth, size_t size,
                     size_t normDetph,
                     double kappa, double alpha, double beta) ;

  /* Specializations: CPU, float */

  template<> vl::Error
  normalize_forward<vl::CPU, float>(float* normalized,
                                    float const* data,
                                    size_t height, size_t width, size_t depth, size_t size,
                                    size_t normDetph,
                                    double kappa, double alpha, double beta) ;

  template<> vl::Error
  normalize_backward<vl::CPU, float>(float* derData,
                                     float const* data,
                                     float const* derNormalized,
                                     size_t height, size_t width, size_t depth, size_t size,
                                     size_t normDetph,
                                     double kappa, double alpha, double beta) ;


  /* Specializations: GPU, float */

#if ENABLE_GPU
  template<> vl::Error
  normalize_forward<vl::GPU, float>(float* normalized,
                                    float const* data,
                                    size_t height, size_t width, size_t depth, size_t size,
                                    size_t normDetph,
                                    double kappa, double alpha, double beta) ;

  template<> vl::Error
  normalize_backward<vl::GPU, float>(float* derData,
                                     float const* data,
                                     float const* derNormalized,
                                     size_t height, size_t width, size_t depth, size_t size,
                                     size_t normDetph,
                                     double kappa, double alpha, double beta) ;
#endif

} }
#endif /* __vl__normalize__ */
