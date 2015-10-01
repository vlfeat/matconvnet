// @file nnbnorm.hpp
// @brief Batch normalizatoion block
// @author Sebastien Ehrhardt
// @author Andrea Vedaldi

/*
Copyright (C) 2015 Sebastien Ehrhardt and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnbnorm__
#define __vl__nnbnorm__

#include "data.hpp"
#include <stdio.h>

namespace vl {

  // This version computes mean and sigma
  vl::Error
  nnbnorm_forward(vl::Context& context,
                  vl::Tensor output,
                  vl::Tensor moments, // [output: can pass null]
                  vl::Tensor data,
                  vl::Tensor filters,
                  vl::Tensor biases,
                  float epsilon) ;

  // This version uses the mean and sigma specified
  vl::Error
  nnbnorm_forward_given_moments(vl::Context& context,
                                vl::Tensor output,
                                vl::Tensor moments, // input
                                vl::Tensor data,
                                vl::Tensor filters,
                                vl::Tensor biases) ;

  vl::Error
  nnbnorm_backward(vl::Context& context,
                   vl::Tensor derData,
                   vl::Tensor derFilters,
                   vl::Tensor derBiaises,
                   vl::Tensor moments,
                   vl::Tensor data,
                   vl::Tensor filters,
                   vl::Tensor biases,
                   vl::Tensor derOutput,
                   float epsilon) ;

  vl::Error
  nnbnorm_backward_given_moments(vl::Context& context,
                                 vl::Tensor derData,
                                 vl::Tensor derFilters,
                                 vl::Tensor derBiaises,
                                 vl::Tensor moments,
                                 vl::Tensor data,
                                 vl::Tensor filters,
                                 vl::Tensor biases,
                                 vl::Tensor derOutput,
                                 float epsilon) ;
}

#endif /* defined(__vl__nnbnorm__) */
