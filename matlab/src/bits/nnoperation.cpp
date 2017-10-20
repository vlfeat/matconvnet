//
//  nnoperation.cpp
//  mcn-lib-cpu
//
//  Created by Andrea Vedaldi on 16/10/2017.
//  Copyright © 2017 Andrea Vedaldi. All rights reserved.
//

#include "nnoperation.hpp"
#include <algorithm>

using namespace vl ;
using namespace vl::nn ;
using namespace std ;

ConvolutionLike::ConvolutionLike(Context& context, Int numSpatialDimensions)
: Operation(context), numSpatialDimensions(numSpatialDimensions)
{
  stride.fill(1) ;
  padding.fill(0) ;
}

vl::ErrorCode ConvolutionLike::setStride(vector<Int> const& stride)
{
  // There must one stride per spatial dimension.
  if (Int(stride.size()) != numSpatialDimensions) {
    return VLE_IllegalArgument ;
  }
  // Stride must be positive.
  if (any_of(begin(stride),begin(stride)+numSpatialDimensions,[](Int x){return x <= 0;})) {
    return VLE_IllegalArgument ;
  }
  copy(begin(stride),begin(stride)+numSpatialDimensions,begin(this->stride)) ;
  return VLE_Success ;
}

vl::ErrorCode ConvolutionLike::setPadding(vector<Int> const& padding)
{
  // There must two padding values per spatial dimension.
  if (Int(padding.size()) != 2*numSpatialDimensions) {
    return VLE_IllegalArgument ;
  }
  // Padding must be non-negative.
  if (any_of(begin(padding),begin(padding)+2*numSpatialDimensions,[](Int x){return x < 0;})) {
    return VLE_IllegalArgument ;
  }
  copy(begin(padding),begin(padding)+2*numSpatialDimensions,begin(this->padding)) ;
  return VLE_Success ;
}

