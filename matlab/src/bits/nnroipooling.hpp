// @file nnroipooling.hpp
// @brief ROI pooling block
// @author Hakan Bilen
// @author Abishek Dutta
// @author Andrea Vedaldi

/*
Copyright (C) 2016-17 Hakan Bilen, Abishek Dutta, and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnroipooling__
#define __vl__nnroipooling__

#include "nnoperation.hpp"
#include <array>

namespace vl { namespace nn {

  class ROIPooling : public Operation {
  public:
    enum Method { Max, Average } ;

    ROIPooling(vl::Context &context,
               std::array<Int,2> subdivisions,
               std::array<double,6> transform,
               Method method) ;

    vl::ErrorCode forward(vl::Tensor &output,
                          vl::Tensor const &input,
                          vl::Tensor const &rois) const ;

    vl::ErrorCode backward(vl::Tensor &derInput,
                           vl::Tensor const &input,
                           vl::Tensor const &rois,
                           vl::Tensor const &derOutput) const ;

    std::array<Int,2> const& getSubdivisions() const {
      return subdivisions ;
    }

    std::array<double,6> const& getTransform() const {
      return transform ;
    }

    Method getMethod() const { return method ; }

  private:
    std::array<Int,2> subdivisions ;
    std::array<double,6> transform ;
    Method method ;
  } ;
  
} }

#endif /* defined(__vl__nnroipooling__) */
