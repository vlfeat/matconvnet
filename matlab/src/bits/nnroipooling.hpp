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
#include <vector>

namespace vl { namespace nn {

  class ROIPooling : public Operation {
  public:
    enum Method { Max, Average } ;

    ROIPooling(Context &context) ;
    ROIPooling(Context &context,
               std::vector<Int> const &subdivisions,
               std::vector<double> const &transform,
               Method method) ;

    ErrorCode forwardShape(TensorShape &output,
                           TensorShape const &input,
                           TensorShape const &rois) const ;

    ErrorCode forward(Tensor &output,
                      Tensor const &input,
                      Tensor const &rois) const ;

    ErrorCode backward(Tensor &derInput,
                       Tensor const &input,
                       Tensor const &rois,
                       Tensor const &derOutput) const ;

    ErrorCode setSubdivisions(std::vector<Int> const &subdivisions) ;
    std::vector<Int> const& getSubdivisions() const {
      return subdivisions ;
    }

    ErrorCode setTransform(std::vector<double> const &transform) ;
    std::vector<double> const& getTransform() const {
      return transform ;
    }

    ErrorCode setMethod(Method method) ;
    Method getMethod() const { return method ; }

    Int getNumSpatialDimensions() const { return 2 ; }

  private:
    std::vector<Int> subdivisions ;
    std::vector<double> transform ;
    Method method ;
  } ;
  
} }

#endif /* defined(__vl__nnroipooling__) */
