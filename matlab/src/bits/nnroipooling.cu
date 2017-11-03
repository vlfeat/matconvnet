// @file nnroipooling.cu
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

#include "nnroipooling.hpp"
#include "impl/dispatcher.hpp"
#include <limits>
#include <cassert>
#include <cstring>
#include <cmath>
#include <iostream>

using namespace vl ;
using namespace vl::nn ;
using namespace vl::impl ;

template<DeviceType deviceType, DataType dataType> struct ROIPoolingForward ;
template<DeviceType deviceType, DataType dataType> struct ROIPoolingBackward ;

#if ENABLE_GPU
#include "nnroipooling_gpu.cu"
#endif

// -------------------------------------------------------------------
//                                                             Helpers
// -------------------------------------------------------------------

template <typename type>
struct acc_max
{
  inline acc_max(Int poolHeight, Int poolWidth, type derOutput = 0)
  :
  value(-std::numeric_limits<type>::infinity()),
  derOutput(derOutput),
  derDataActivePt(NULL)
  { }

  inline void accumulate_forward(type x) {
    value = std::max(value, x) ;
  }

  inline void accumulate_backward(type const* data, type* derDataPt) {
    type x = *data ;
    if (x > value) {
      value = x ;
      derDataActivePt = derDataPt ;
    }
  }

  inline type done_forward() const {
    return value ;
  }

  inline void done_backward() const {
    if (derDataActivePt) { *derDataActivePt += derOutput ; }
  }

  type value ;
  type derOutput ;
  type* derDataActivePt ;
} ;

template <typename type>
struct acc_sum
{
  inline acc_sum(Int poolHeight, Int poolWidth, type derOutput = 0)
  :
  value(0),
  scale(type(1)/type(poolHeight*poolWidth)),
  derOutput(derOutput)
  { }

  inline void accumulate_forward(type x) {
    value += x ;
  }

  inline void accumulate_backward(type const* data, type* derDataPt) {
    *derDataPt += derOutput * scale ;
  }

  inline type done_forward() const {
    return value * scale ;
  }

  inline void done_backward() const { }

  type value ;
  type derOutput ;
  type scale;
} ;

// -------------------------------------------------------------------
//                                                             Forward
// -------------------------------------------------------------------

template<DataType dataType, class Accumulator>
struct ROIPoolingForwardCPU
{
  vl::ErrorCode operator()(ROIPooling const &op,
                           Tensor &pooled,
                           Tensor const &input,
                           Tensor const &rois)
  {
    static const std::string signature = std::string("ROIPoolingForward[MCN,")
    + DeviceTypeTraits<VLDT_CPU>::name + "," + DataTypeTraits<dataType>::name + "]" ;
    VLLOG(op,1) << signature ;

    typedef typename vl::DataTypeTraits<dataType>::type type ;
    Int numROIs = rois.getNumElements() / 5 ;
    Int height = input.getHeight() ;
    Int width = input.getWidth() ;
    Int depth = input.getNumChannels() ;
    Int size = input.getCardinality() ;
    auto roisData = (type const*)rois.getMemory() ;
    auto inputData = (type const*)input.getMemory() ;
    auto pooledData = (type*)pooled.getMemory() ;

    // For each ROI R = [t x1 y1 x2 y2].
    for (Int roi = 0; roi < numROIs; ++roi) {

      // Apply scale and offset to each ROI coordinate.
      type u1_ = roisData[5 * roi + 1] ;
      type v1_ = roisData[5 * roi + 2] ;
      type u2_ = roisData[5 * roi + 3] ;
      type v2_ = roisData[5 * roi + 4] ;

      auto const& tf = op.getTransform() ;
      type u1 = (type)tf[0] * u1_ + (type)tf[2] * v1_ + (type)tf[4] ;
      type v1 = (type)tf[1] * u1_ + (type)tf[3] * v1_ + (type)tf[5] ;
      type u2 = (type)tf[0] * u2_ + (type)tf[2] * v2_ + (type)tf[4] ;
      type v2 = (type)tf[1] * u2_ + (type)tf[3] * v2_ + (type)tf[5] ;

      // First and last pixel of each ROI (rounded
      // for compatibility with the Caffe definition).
      Int roi_image   = (Int)roisData[5 * roi + 0];
      Int roi_start_h = (Int)::round(v1) - 1 ;
      Int roi_start_w = (Int)::round(u1) - 1 ;
      Int roi_end_h   = (Int)::round(v2) - 1 ;
      Int roi_end_w   = (Int)::round(u2) - 1 ;
      Int roi_height  = std::max(roi_end_h - roi_start_h + 1, (Int)1) ;
      Int roi_width   = std::max(roi_end_w - roi_start_w + 1, (Int)1) ;

      roi_image = std::min(std::max(roi_image - 1, (Int)0), size - 1) ;
      type const * data_offset = inputData + (roi_image * depth) * (width*height) ;

      type bin_size_h = (type)roi_height / op.getSubdivisions()[0] ;
      type bin_size_w = (type)roi_width / op.getSubdivisions()[1] ;

      // For each feature channel.
      for (Int z = 0; z < depth; ++z) {

        // For each column of tiles.
        for (Int pw = 0; pw < op.getSubdivisions()[1]; ++pw) {
          Int wstart = (Int)floor(((type)pw) * bin_size_w) ;
          Int wend = (Int)ceil(((type)(pw + 1)) * bin_size_w) ;
          wstart = std::min(std::max(wstart + roi_start_w, (Int)0), width) ;
          wend = std::min(std::max(wend + roi_start_w, (Int)0), width) ;

          // For each tile in a column.
          for (Int ph = 0; ph < op.getSubdivisions()[0]; ++ph) {
            Int hstart = (Int)floor(((type)ph) * bin_size_h) ;
            Int hend = (Int)ceil(((type)(ph + 1)) * bin_size_h) ;
            hstart = std::min(std::max(hstart + roi_start_h, (Int)0), height) ;
            hend = std::min(std::max(hend + roi_start_h, (Int)0), height) ;

            bool is_empty = (hend <= hstart) || (wend <= wstart);

            if (is_empty) {
              *pooledData++ = 0 ;
            }
            else {
              Accumulator acc(hend - hstart, wend - wstart) ;
              for (Int w = wstart ; w < wend; ++w) {
                for (Int h = hstart ; h < hend; ++h) {
                  auto const index = w * height + h ;
                  acc.accumulate_forward(data_offset[index]) ;
                }
              }
              *pooledData++ = acc.done_forward() ;
            }
          } // end of ph
        } // end of pw
        data_offset += width*height;
      } // end of z
    } // end of n
    return VLE_Success ;
  }
} ;

template<DataType dataType>
struct ROIPoolingForward<VLDT_CPU,dataType>
{
  vl::ErrorCode operator()(ROIPooling const &op,
                           Tensor &pooled,
                           Tensor const &input,
                           Tensor const &rois)
  {
    switch (op.getMethod()) {
      case ROIPooling::Max:
        return
        ROIPoolingForwardCPU<dataType,acc_max<typename vl::DataTypeTraits<dataType>::type> >
        ()(op,pooled,input,rois) ;
      case ROIPooling::Average:
        return
        ROIPoolingForwardCPU<dataType,acc_sum<typename vl::DataTypeTraits<dataType>::type> >
        ()(op,pooled,input,rois) ;
      default: return VLE_IllegalArgument ;
    }
  }
} ;

// -------------------------------------------------------------------
//                                                            Backward
// -------------------------------------------------------------------

template<DataType dataType, class Accumulator>
struct ROIPoolingBackwardCPU
{
  vl::ErrorCode operator()(ROIPooling const &op,
                           Tensor &derInput,
                           Tensor const &input,
                           Tensor const &rois,
                           Tensor const &derOutput)
  {
    static const std::string signature = std::string("ROIPoolingBackward[MCN,")
    + DeviceTypeTraits<VLDT_CPU>::name + "," + DataTypeTraits<dataType>::name + "]" ;
    VLLOG(op,1) << signature ;

    typedef typename vl::DataTypeTraits<dataType>::type type ;
    Int numROIs = rois.getNumElements() / 5 ;
    Int height = input.getHeight() ;
    Int width = input.getWidth() ;
    Int depth = input.getNumChannels() ;
    Int size = input.getCardinality() ;
    
    auto derInputData = (type*)derInput.getMemory() ;
    auto roisData = (type const*)rois.getMemory() ;
    auto inputData = (type const*)input.getMemory() ;
    auto derOutputData = (type const*)derOutput.getMemory() ;

    memset(derInputData, 0, (size_t)derInput.getNumElements() * sizeof(type)) ;

    // For each ROI R = [t x1 y1 x2 y2].
    for (Int roi = 0; roi < numROIs ; ++roi) {

      // Apply sacle and offset to each ROI coordinate.
      type u1_ = roisData[5 * roi + 1] ;
      type v1_ = roisData[5 * roi + 2] ;
      type u2_ = roisData[5 * roi + 3] ;
      type v2_ = roisData[5 * roi + 4] ;

      auto const& tf = op.getTransform() ;
      type u1 = (type)tf[0] * u1_ + (type)tf[2] * v1_ + (type)tf[4] ;
      type v1 = (type)tf[1] * u1_ + (type)tf[3] * v1_ + (type)tf[5] ;
      type u2 = (type)tf[0] * u2_ + (type)tf[2] * v2_ + (type)tf[4] ;
      type v2 = (type)tf[1] * u2_ + (type)tf[3] * v2_ + (type)tf[5] ;

      // First and last pixel of each ROI (rounded
      // for compatibility with the Caffe definition).
      Int roi_image   = (Int)roisData[5 * roi + 0];
      Int roi_start_h = (Int)::round(v1) - 1 ;
      Int roi_start_w = (Int)::round(u1) - 1 ;
      Int roi_end_h   = (Int)::round(v2) - 1 ;
      Int roi_end_w   = (Int)::round(u2) - 1 ;
      Int roi_height = std::max(roi_end_h - roi_start_h + 1, (Int)1) ;
      Int roi_width = std::max(roi_end_w - roi_start_w + 1, (Int)1) ;

      roi_image = std::min(std::max(roi_image - 1,(Int)0), size - 1) ;
      type const * data_offset = inputData + roi_image * (depth*width*height) ;
      type * derInputData_offset = derInputData + roi_image * (depth*width*height) ;

      const type bin_size_h = (type)roi_height / op.getSubdivisions()[0] ;
      const type bin_size_w = (type)roi_width / op.getSubdivisions()[1] ;

      // For each feature channel.
      for (Int z = 0; z < depth; ++z) {

        // For each column of tiles.
        for (Int pw = 0; pw < op.getSubdivisions()[1]; ++pw) {
          Int wstart = (Int)floor(((type)pw) * bin_size_w) ;
          Int wend = (Int)ceil(((type)(pw + 1)) * bin_size_w) ;
          wstart = std::min(std::max(wstart + roi_start_w, (Int)0), width) ;
          wend = std::min(std::max(wend + roi_start_w, (Int)0), width) ;

          // For each tile in a column.
          for (Int ph = 0; ph < op.getSubdivisions()[0]; ++ph) {
            Int hstart = (Int)floor(((type)ph) * bin_size_h) ;
            Int hend = (Int)ceil(((type)(ph + 1)) * bin_size_h) ;
            hstart = std::min(std::max(hstart + roi_start_h, (Int)0), height) ;
            hend = std::min(std::max(hend + roi_start_h, (Int)0), height) ;

            Accumulator acc(hend - hstart, wend - wstart, *derOutputData++) ;
            for (Int w = wstart; w < wend; ++w) {
              for (Int h = hstart; h < hend; ++h) {
                auto const index = w * height + h ;
                acc.accumulate_backward(&data_offset[index],
                                        &derInputData_offset[index]) ;
              }
            }
            acc.done_backward() ;
          } // end of pw
        } // end of ph
        data_offset += width*height ;
        derInputData_offset += width*height ;
      } // end of z
    } // end of n

    return VLE_Success ;
  }
} ;

template<DataType dataType>
struct ROIPoolingBackward<VLDT_CPU,dataType>
{
  vl::ErrorCode operator()(ROIPooling const&op,
                           Tensor &derInput,
                           Tensor const &input,
                           Tensor const &rois,
                           Tensor const &derOutput)
  {
    switch (op.getMethod()) {
      case ROIPooling::Max: return
        ROIPoolingBackwardCPU<dataType,acc_max<typename vl::DataTypeTraits<dataType>::type> >
        ()(op,derInput,input,rois,derOutput) ;
      case ROIPooling::Average: return
        ROIPoolingBackwardCPU<dataType,acc_sum<typename vl::DataTypeTraits<dataType>::type> >
        ()(op,derInput,input,rois,derOutput) ;
      default: return VLE_IllegalArgument ;
    }
  }
} ;

// -------------------------------------------------------------------
/// MARK: - Driver
// -------------------------------------------------------------------

ROIPooling::ROIPooling(Context &context,
                       std::vector<Int> const& subdivisions,
                       std::vector<double> const& transform,
                       Method method) :
Operation(context),
subdivisions(subdivisions),
transform(transform),
method(method)
{ }

ROIPooling::ROIPooling(Context &context)
:
Operation(context),
subdivisions {1,1},
transform {1., 0., 0., 1., 0., 0.},
method (Max)
{ }

ErrorCode ROIPooling::setSubdivisions(std::vector<Int> const& subdivisions) {
  // Stride must be positive.
  if (any_of(begin(subdivisions),end(subdivisions),[](Int x){return x <= 0;})) {
    return getContext().setError
    (VLE_IllegalArgument, "An element of SUBDIVISIONS is less than 1.") ;
  }
  // There must one stride per spatial dimension.
  if (Int(subdivisions.size()) == getNumSpatialDimensions()) {
    this->subdivisions = subdivisions ;
  }
  else if (subdivisions.size() == 1) {
    fill(begin(this->subdivisions),end(this->subdivisions),subdivisions[0]) ;
  }
  else {
    return getContext().setError
    (VLE_IllegalArgument, "SUBDIVISIONS is neither scalar nor has the same"
     " cardinality as the number of spatial dimensions.") ;
  }
  return VLE_Success ;
}

ErrorCode ROIPooling::setTransform(std::vector<double> const& transform)
{
  // There must one stride per spatial dimension.
  Int ns = getNumSpatialDimensions() ;
  if (Int(transform.size()) == (ns)*(ns+1)) {
    this->transform = transform ;
  }
  else if ((Int)transform.size() == 2*ns) {
    fill(begin(this->transform),end(this->transform),.0) ;
    for (Int i = 0 ; i < ns ; ++i) {
      this->transform[size_t(i + ns*i)] = transform[size_t(i)] ;
      this->transform[size_t(i + ns*ns)] = transform[size_t(i+ns)] ;
    }
  }
  else if (transform.size() == 1) {
    fill(begin(this->transform),end(this->transform),.0) ;
    for (Int i = 0 ; i < ns ; ++i) {
      this->transform[size_t(i + ns*i)] = transform[0] ;
    }
  }
  else {
    return getContext().setError
    (VLE_IllegalArgument, "TRANSFORMS is neither scalar nor has the the "
     "appropriate size for the number of spatial dimensions.") ;
  }
  return VLE_Success ;
}

ErrorCode ROIPooling::setMethod(Method method) {
  if (method != Average && method != Max) {
    return getContext().setError(VLE_IllegalArgument, "Unknown METHOD.") ;
  }
  this->method = method ;
  return VLE_Success ;
}

vl::ErrorCode
ROIPooling::forwardShape(TensorShape &output,
                         TensorShape const& input,
                         TensorShape const& rois) const
{
  output.clear() ;
  auto ns = getNumSpatialDimensions() ;

  // INPUT must have spatial dimensions, channels, and instances.
  if (input.getNumDimensions() > ns+2) {
    return getContext().setError
    (VLE_TensorShapeMismatch, "ROIPooling: INPUT has too many dimensions.") ;
  }

  // ROIS must contain an integer number of ROI specifications.
  Int numROIs = rois.getNumElements() / 5 ;
  if (numROIs * 5 != rois.getNumElements()) {
    return getContext().setError
    (VLE_TensorShapeMismatch, "ROIPooling: the number of elements of ROI is not a multiple of 5.") ;
  }

  // Output has size SUBD... x INPUT_CHANNELS x NUMROIS.
  std::vector<Int> dims (size_t(ns + 2)) ;
  copy(begin(subdivisions),end(subdivisions),begin(dims)) ;
  dims[size_t(ns)] = input.getDimension(ns) ;
  dims[size_t(ns)+1] = numROIs ;

  output = dims ;
  return VLE_Success ;
}

vl::ErrorCode
ROIPooling::forward(Tensor &output,
                    Tensor const &input,
                    Tensor const &rois) const
{
  // Validate arguments.
  ErrorCode error ;
  if (!check_tensor_compatibility(output,input,rois)) {
    return getContext().setError
    (VLE_IllegalArgument,
     "ROIPoolingForward: the tensors have mismatching data or device type.") ;
  }
  TensorShape outputShape ;
  if ((error = forwardShape(outputShape, input, rois)) != VLE_Success) {
    return error ;
  }
  if (output != outputShape) {
    return getContext().setError
    (VLE_TensorShapeMismatch,
     "ROIPoolingForward: OUTPUT does not have the appropriate dimensions.") ;
  }
  if (input.isEmpty() || input.isNull()) {
    return getContext().setError
    (VLE_IllegalArgument,
     "ROIPoolingForward: INPUT is empty or null.") ;
  }
  if (input.isEmpty() || input.isNull()) {
    return getContext().setError
    (VLE_IllegalArgument,
     "ROIPoolingForward: OUTPUT is empty or null.") ;
  }
  if (rois.isEmpty() || rois.isNull()) {
    return getContext().setError
    (VLE_IllegalArgument,
     "ROIPoolingForward: ROI is empty or null.") ;
  }

  VLLOG(*this,1)
  << "ROIPoolingForward:"
  << " subdivisions=" << pretty(getSubdivisions())
  << " transform=" << pretty(getTransform()) ;

  VLLOG(*this,1)
  << "ROIPoolingForward:"
  << " input=" << pretty(input.getDimensions())
  << " rois=" << pretty(rois.getDimensions())
  << " output=" << pretty(output.getDimensions()) ;

  return getContext().passError
  (dispatch<ROIPoolingForward>()(*this,output,input,rois),
   "ROIPoolingForward") ;
}

vl::ErrorCode
ROIPooling::backward(Tensor &derInput,
                     Tensor const &input,
                     Tensor const &rois,
                     Tensor const &derOutput) const
{
  // Validate arguments.
  ErrorCode error ;
  if (!check_tensor_compatibility(derInput,input,rois,derOutput)) {
    return getContext().setError
    (VLE_IllegalArgument,
     "ROIPoolingForward: the tensors have mismatching data or device type.") ;
  }
  TensorShape outputShape ;
  if ((error = forwardShape(outputShape, input, rois)) != VLE_Success) {
    return error ;
  }
  if (derOutput != outputShape) {
    return getContext().setError
    (VLE_TensorShapeMismatch,
     "ROIPoolingForward: OUTPUT does not have the appropriate dimensions.") ;
  }
  if (input.isEmpty() || input.isNull()) {
    return getContext().setError
    (VLE_IllegalArgument,
     "ROIPoolingForward: INPUT is empty or null.") ;
  }
  if (input.isEmpty() || input.isNull()) {
    return getContext().setError
    (VLE_IllegalArgument,
     "ROIPoolingForward: OUTPUT is empty or null.") ;
  }
  if (rois.isEmpty() || rois.isNull()) {
    return getContext().setError
    (VLE_IllegalArgument,
     "ROIPoolingForward: ROI is empty or null.") ;
  }

  VLLOG(*this,1)
  << "ROIPoolingBackward:"
  << " subdivisions=" << pretty(getSubdivisions())
  << " transform=" << pretty(getTransform())
  << " method=" << (getMethod() == Average ? "Average" : "Max") ;

  VLLOG(*this,1)
  << "ROIPoolingBackward:"
  << " derInput=" << pretty(derInput.getDimensions())
  << " input=" << pretty(input.getDimensions())
  << " rois=" << pretty(rois.getDimensions())
  << " derOutput=" << pretty(derOutput.getDimensions()) ;

  return getContext().passError
  (dispatch<ROIPoolingBackward>()(*this,derInput,input,rois,derOutput),
   "ROIPoolingBackward") ;
}





