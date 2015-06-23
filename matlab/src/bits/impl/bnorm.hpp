// @file bnorm.hpp
// @brief Batch Normalization block implementation
// @author Sebastien Ehrhardt

/*
Copyright (C) 2015 Sebastien Ehrhardt.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl_bnorm__
#define __vl_bnorm__

#include "../data.hpp"
#include <cstddef>
//TODO derOutput is const ?

namespace vl { namespace impl {

  template<vl::Device dev, typename type> vl::Error
  bnorm_forward(Context& context, type* output,
                type const* data,
                type* filters,
                type* biaises,
                size_t height, size_t width, size_t depth, size_t size, type epsilon) ;


  template<vl::Device dev, typename type> vl::Error
  bnorm_backward(Context& context, type* derData,
  	 	 type* derFilters,
  	 	 type* derBiaises,
                 type const* data,
                 type const* filters,
                 type const* biaises,
                 size_t height, size_t width, size_t depth, size_t size,
                 type* derOutput, float epsilon) ;


  /* Specializations: CPU, float */

  template<> vl::Error
  bnorm_forward<vl::CPU, float>(Context& context, float* output,
                                float const* data,
                                float* filters,
                                float* biaises,
                                size_t height, size_t width, size_t depth, size_t size, float epsilon) ;

  template<> vl::Error
  bnorm_backward<vl::CPU, float>(Context& context, float* derData,
  	 	                 float* derFilters,
  	 	                 float* derBiaises,
                                 float const* data,
                                 float const* filters,
                                 float const* biaises,
                                 size_t height, size_t width, size_t depth, size_t size,
                                 float* derOutput, float epsilon)  ;

  /* Specializations: GPU, float */

  #if ENABLE_GPU

  template<> vl::Error
  bnorm_forward<vl::GPU, float>(Context& context, float* output,
                                float const* data,
                                float* filters,
                                float* biaises,
                                size_t height, size_t width, size_t depth, size_t size, float epsilon) ;

  template<> vl::Error
  bnorm_backward<vl::GPU, float>(Context& context, float* derData,
  	 	                 float* derFilters,
  	 	                 float* derBiaises,
                                 float const* data,
                                 float const* filters,
                                 float const* biaises,
                                 size_t height, size_t width, size_t depth, size_t size,
                                 float* derOutput, float epsilon) ;


  #endif

} }
#endif /* __vl_bnorm__ */
