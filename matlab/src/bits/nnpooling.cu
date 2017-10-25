// @file nnpooling.cu
// @brief Pooling block
// @author Andrea Vedaldi

/*
Copyright (C) 2014-17 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnpooling.hpp"
#include "impl/dispatcher.hpp"
#include <cassert>
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace vl ;
using namespace vl::nn ;
using namespace vl::impl ;
//using namespace std ;

template<DeviceType deviceType, DataType dataType> struct PoolingForward ;
template<DeviceType deviceType, DataType dataType> struct PoolingBackward ;
template<DataType dataType> struct PoolingForwardCudnn ;
template<DataType dataType> struct PoolingBackwardCudnn ;

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

  /* note: data is unused */
  inline void accumulate_backward(type const* data, type* derDataPt) {
    *derDataPt += derOutput * scale ;
  }

  inline type done_forward() const {
    return value * scale ;
  }

  inline void done_backward() const { }

  type value ;
  type derOutput ;
  type scale ;
} ;

// -------------------------------------------------------------------
//                                                             Forward
// -------------------------------------------------------------------

template<DataType dataType, class Accumulator>
struct PoolingForwardCPU
{
  vl::ErrorCode operator()(Pooling const &op,
                           Tensor &output,
                           Tensor const &input)
  {
    typedef typename vl::DataTypeTraits<dataType>::type type ;
    Int height = input.getHeight() ;
    Int width = input.getWidth() ;
    Int depth = input.getNumChannels() ;
    Int size = input.getCardinality() ;
    auto inputData = (type const*)input.getMemory() ;
    auto outputData = (type*)output.getMemory() ;

    TensorShape outputShape ;
    op.forwardShape(outputShape, input) ;
    assert(output == outputShape) ;
    Int outputHeight = outputShape.getDimension(0) ;
    Int outputWidth = outputShape.getDimension(1) ;

    for (Int z = 0; z < depth * size ; ++z) {
      for (Int x = 0; x < outputWidth ; ++x) {
        for (Int y = 0; y < outputHeight ; ++y) {
          Int y1 = y * op.getStride(0) - op.getPadding(0) ;
          Int x1 = x * op.getStride(1) - op.getPadding(2) ;
          Int y2 = std::min(y1 + op.getShape(0), height) ;
          Int x2 = std::min(x1 + op.getShape(1), width) ;
          y1 = std::max(y1, (Int)0) ;
          x1 = std::max(x1, (Int)0) ;
          Accumulator acc(y2 - y1, x2 - x1) ;
          for (Int u = x1 ; u < x2 ; ++u) {
            for (Int v = y1 ; v < y2 ; ++v) {
              acc.accumulate_forward(inputData[u * height + v]) ;
            }
          }
          outputData[x * outputHeight + y] = acc.done_forward() ;
        }
      }
      inputData += width*height ;
      outputData += outputWidth*outputHeight ;
    }
    return VLE_Success ;
  }
} ;

template<DataType dataType>
struct PoolingForward<VLDT_CPU,dataType>
{
  vl::ErrorCode operator()(Pooling const&op,
                           Tensor output,
                           Tensor input)
  {
    switch (op.getMethod()) {
      case Pooling::Max:
        return
        PoolingForwardCPU<dataType,acc_max<typename vl::DataTypeTraits<dataType>::type> >
        ()(op,output,input) ;
      case Pooling::Average:
        return
        PoolingForwardCPU<dataType,acc_sum<typename vl::DataTypeTraits<dataType>::type> >
        ()(op,output,input) ;
      default:
        return VLE_IllegalArgument ;
    }
  }
} ;

// -------------------------------------------------------------------
//                                                            Backward
// -------------------------------------------------------------------

template<DataType dataType, class Accumulator>
struct PoolingBackwardCPU
{
  vl::ErrorCode operator()(Pooling const &op,
                           Tensor &derInput,
                           Tensor const &input,
                           Tensor const &derOutput)
  {
    typedef typename vl::DataTypeTraits<dataType>::type type ;
    auto height = input.getHeight() ;
    auto width = input.getWidth() ;
    auto depth = input.getNumChannels() ;
    auto size = input.getCardinality() ;
    auto derInputData = (type*)derInput.getMemory() ;
    auto inputData = (type const*)input.getMemory() ;
    auto derOutputData = (type const*)derOutput.getMemory() ;

    TensorShape outputShape ;
    op.forwardShape(outputShape, input) ;
    auto outputHeight = outputShape.getDimension(0) ;
    auto outputWidth = outputShape.getDimension(1) ;

    for (Int z = 0; z < depth * size ; ++z) {
      for (Int x = 0; x < outputWidth ; ++x) {
        for (Int y = 0; y < outputHeight ; ++y) {
          Int y1 = y * op.getStride(0) - op.getPadding(0) ;
          Int x1 = x * op.getStride(1) - op.getPadding(2) ;
          Int y2 = std::min(y1 + op.getShape(0), height) ;
          Int x2 = std::min(x1 + op.getShape(1), width) ;
          y1 = std::max(y1, (Int)0) ;
          x1 = std::max(x1, (Int)0) ;
          Accumulator acc(y2 - y1, x2 - x1, derOutputData[x * outputHeight + y]) ;
          for (Int u = x1 ; u < x2 ; ++u) {
            for (Int v = y1 ; v < y2 ; ++v) {
              acc.accumulate_backward(&inputData[u * height + v],
                                      &derInputData[u * height + v]) ;
            }
          }
          acc.done_backward() ;
        }
      }
      inputData += width*height ;
      derInputData += width*height ;
      derOutputData += outputWidth*outputHeight ;
    }
    return VLE_Success ;
  }
} ;

template<DataType dataType>
struct PoolingBackward<VLDT_CPU,dataType>
{
  vl::ErrorCode operator()(Pooling const &op,
                           Tensor &derInput,
                           Tensor const &input,
                           Tensor const &derOutput)
  {
    switch (op.getMethod()) {
      case Pooling::Max:
        return
        PoolingBackwardCPU<dataType,acc_max<typename vl::DataTypeTraits<dataType>::type> >
        ()(op,derInput,input,derOutput) ;
      case Pooling::Average:
        return
        PoolingBackwardCPU<dataType,acc_sum<typename vl::DataTypeTraits<dataType>::type> >
        ()(op,derInput,input,derOutput) ;
      default:
        return VLE_IllegalArgument ;
    }
  }
} ;

// -------------------------------------------------------------------
//                                                              Driver
// -------------------------------------------------------------------

#if ENABLE_GPU
#include "nnpooling_gpu.cu"
#endif

#if ENABLE_CUDNN
#include "nnpooling_cudnn.cu"
#endif

Pooling::Pooling(Context &context,
                 Int poolHeight, Int poolWidth,
                 Int strideY, Int strideX,
                 Int padTop, Int padBottom,
                 Int padLeft, Int padRight,
                 Method method)
: ConvolutionLike(context,2),
  method(method)
{
  setShape({poolHeight,poolWidth}) ;
  setStride({strideY,strideX}) ;
  setPadding({padTop,padBottom,padLeft,padRight}) ;
}

Pooling::Pooling(Context &context)
: ConvolutionLike(context), method(Average)
{
  shape.fill(1) ;
}

ErrorCode Pooling::setMethod(Method method) {
  if (method != Average && method != Max) {
    return VLE_IllegalArgument ;
  }
  this->method = method ;
  return VLE_Success ;
}

ErrorCode Pooling::setShape(std::vector<Int> const& shape) {
  // There must one shape dimension per spatial dimension.
  if ((Int)shape.size() != getNumSpatialDimensions()) {
    return VLE_IllegalArgument ;
  }
  // Shape must be positive.
  if (any_of(begin(shape),begin(shape)+getNumSpatialDimensions(),[](Int x){return x <= 0;})) {
    return VLE_IllegalArgument ;
  }
  copy(begin(shape),begin(shape)+getNumSpatialDimensions(),begin(this->shape)) ;
  return VLE_Success ;
}

vl::ErrorCode
Pooling::forward(Tensor &output,
                 Tensor const &input) const
{
  return dispatch_cudnn<PoolingForward,PoolingForwardCudnn>()(*this,output,input) ;
}

vl::ErrorCode
Pooling::forwardShape(TensorShape& output,
                      TensorShape const& input) const
{
  output = TensorShape() ; // null
  if (input.getNumDimensions() < getNumSpatialDimensions()) {
    return VLE_IllegalArgument ;
  }
  output = input ;
  for (Int d = 0 ; d < getNumSpatialDimensions() ; ++d) {
    auto odim = convLikeSizeHelper(input.getDimension(d),
                                   getShape(d),
                                   getStride(d),
                                   {getPadding(2*d),getPadding(2*d+1)},
                                   1) ;
    output.setDimension(d, odim) ;
  }
  return VLE_Success ;
}

vl::ErrorCode
Pooling::backward(Tensor &derInput,
                  Tensor const &input,
                  Tensor const &derOutput) const
{
  return dispatch_cudnn<PoolingBackward,PoolingBackwardCudnn>()(*this,derInput,input,derOutput) ;
}
