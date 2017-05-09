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

#include "data.hpp"
#include <array>

namespace vl { namespace nn {

  class ROIPooling {
  public:
    enum Method { Max, Average } ;

    ROIPooling(vl::Context &context,
               std::array<int,2> subdivisions,
               std::array<double,6> transform,
               Method method) ;

    vl::ErrorCode forward(vl::Tensor output,
                          vl::Tensor input,
                          vl::Tensor rois) ;

    vl::ErrorCode backward(vl::Tensor derInput,
                           vl::Tensor input,
                           vl::Tensor rois,
                           vl::Tensor derOutput) ;

    vl::Context& context ;
    std::array<int,2> subdivisions ;
    std::array<double,6> transform ;
    Method method ;
  } ;
  
} }

namespace vl {
  enum ROIPoolingMethod { vlROIPoolingMax, vlROIPoolingAverage } ;

  vl::ErrorCode
  nnroipooling_forward(vl::Context& context,
                       vl::Tensor output,
                       vl::Tensor data,
                       vl::Tensor rois,
                       ROIPoolingMethod method,
                       int const subdivisions[2],
                       double const transform[6]) ;

  vl::ErrorCode
  nnroipooling_backward(vl::Context& context,
                        vl::Tensor derData,
                        vl::Tensor data,
                        vl::Tensor rois,
                        vl::Tensor derOutput,
                        ROIPoolingMethod method,
                        int const subdivisions[2],
                        double const transform[6]) ;
}

#endif /* defined(__vl__nnroipooling__) */
