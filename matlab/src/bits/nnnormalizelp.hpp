// @file nnnormalizelp.hpp
// @brief Batch normalizatoion block
// @author Sebastien Ehrhardt
// @author Andrea Vedaldi

/*
Copyright (C) 2017 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnnormalizelp__
#define __vl__nnnormalizelp__

#include "data.hpp"
#include <stdio.h>
#include <vector>

namespace vl { namespace nn {

  class NormalizeLp {
  public:
    NormalizeLp(vl::Context &context,
                std::vector<int> const& selectedDimensions,
                double exponent = 2.0,
                double epsilon = 1e-3) ;

    vl::TensorShape getNormsShapeForData(vl::Tensor data) ;

    vl::ErrorCode forward(vl::Tensor output,
                          vl::Tensor norms, // [output: can pass null]
                          vl::Tensor data) ;

    vl::ErrorCode forwardWithNorms(vl::Tensor output,
                                     vl::Tensor norms,
                                     vl::Tensor data) ;

    vl::ErrorCode backward(vl::Tensor derData,
                           vl::Tensor moments,
                           vl::Tensor data,
                           vl::Tensor derOutput) ;

    vl::ErrorCode backwardWithNorms(vl::Tensor derData,
                                      vl::Tensor norms,
                                      vl::Tensor data,
                                      vl::Tensor derOutput) ;

    vl::Context& context ;
    std::vector<int> selectedDimensions ;
    double exponent ;
    double epsilon ;
  } ;

} }

#endif /* defined(__vl__nnnormalizelp__) */
