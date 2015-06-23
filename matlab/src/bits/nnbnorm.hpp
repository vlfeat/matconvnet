// @file nnbnorm.hpp
// @brief Batch normalizatoion block
// @author Sebastien Ehrhardt

/*
Copyright (C) 2015 Sebastien Ehrhardt.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnbnorm__
#define __vl__nnbnorm__

#include "data.hpp"
#include <stdio.h>

namespace vl {

  vl::Error
  nnbnorm_forward(vl::Context& context,
                  vl::Tensor output,
                  vl::Tensor data,
                  vl::Tensor filters,
                  vl::Tensor biases,
                  float epsilon) ;

  vl::Error
  nnbnorm_backward(vl::Context& context,
                   vl::Tensor derData,
                   vl::Tensor derFilters,
                   vl::Tensor derBiaises,
                   vl::Tensor data,
                   vl::Tensor filters,
                   vl::Tensor biases,
                   vl::Tensor derOutput,
                   float epsilon) ;
}

#endif /* defined(__vl__nnbnorm__) */
