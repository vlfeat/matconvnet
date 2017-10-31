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
  static char const* name ;

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
  static char const* name ;

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

template <vl::nn::Pooling::Method method> struct pooling_method_traits {  } ;
template <> struct pooling_method_traits<vl::nn::Pooling::Average> { static char const* name ; }  ;
template <> struct pooling_method_traits<vl::nn::Pooling::Max> { static char const* name ; }  ;
char const *  pooling_method_traits<vl::nn::Pooling::Average>::name = "average" ;
char const *  pooling_method_traits<vl::nn::Pooling::Max>::name = "max" ;

template <typename type> const char * acc_max<type>::name = "max" ;
template <typename type> const char * acc_sum<type>::name = "average" ;

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
    static const std::string signature = std::string("PoolingForward[MCN,")
    + Accumulator::name + ","
    + DeviceTypeTraits<VLDT_CPU>::name + "," + DataTypeTraits<dataType>::name + "]" ;

    VLLOG(op,1) << signature ;

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
    static const std::string signature = std::string("PoolingBackward[MCN,")
    + Accumulator::name + ","
    + DeviceTypeTraits<VLDT_CPU>::name + "," + DataTypeTraits<dataType>::name + "]" ;

    VLLOG(op,1) << signature ;

    typedef typename vl::DataTypeTraits<dataType>::type type ;
    Int height = input.getHeight() ;
    Int width = input.getWidth() ;
    Int depth = input.getNumChannels() ;
    Int size = input.getCardinality() ;

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
: ConvolutionLike(context,2), method(Average),
shape((size_t)getNumSpatialDimensions(),1)
{ }

ErrorCode Pooling::setMethod(Method method) {
  if (method != Average && method != Max) {
    return getContext().setError(VLE_IllegalArgument,
                                 "Pooling: Uknown pooling method") ;
  }
  this->method = method ;
  return VLE_Success ;
}

vl::ErrorCode Pooling::setShape(std::vector<Int> const& shape)
{
  // Shape must be positive.
  if (any_of(begin(shape),end(shape),[](Int x){return x <= 0;})) {
    return getContext().setError
    (VLE_IllegalArgument, "An element of SHAPE is less than 1.") ;
  }
  // There must one shape per spatial dimension.
  if (Int(shape.size()) == getNumSpatialDimensions()) {
    this->shape = shape ;
  }
  else if (shape.size() == 1) {
    fill(begin(this->shape),end(this->shape),shape[0]) ;
  }
  else {
    return getContext().setError
    (VLE_IllegalArgument, "SHAPE is neither scalar nor has the same"
     " cardinality as the number of spatial dimensions.") ;
  }
  return VLE_Success ;
}

vl::ErrorCode
Pooling::forwardShape(TensorShape& output,
                      TensorShape const& input) const
{
  output.clear() ;

  // The input tensor cannot be empty.
  if (input.isEmpty()) {
    return getContext().setError
    (VLE_TensorShapeMismatch, "PoolingForwardShape: INPUT is empty.") ;
  }

  // We pretend all tensors have an infinite number of dimensions,
  // potentially singleton.
  Int ns = getNumSpatialDimensions() ;

  // The tensors should have ns+2 dimensions. Todo: we may relax that by implicitly
  // folding the excess dimensions.
  if (input.getNumDimensions() > ns + 2) {
    return getContext().setError(VLE_TensorShapeMismatch,
                                 "PoolingForwardShape: INPUT has too many dimensions.") ;
  }

  for (Int d = 0 ; d < ns ; ++d) {
    auto odim = convLikeSizeHelper(input.getDimension(d),
                                   getShape(d),
                                   getStride(d),
                                   {getPadding(2*d),getPadding(2*d+1)},
                                   1) ;
    if (odim <= 0) {
      output.clear() ;
      return getContext().setError
      (VLE_TensorShapeMismatch,
       "PoolingForwardShape: the spatial dimensions of INPUT are too small for FILTER and the convolution parameters.") ;
    }
    if (getShape(d) <= getPadding(2*d) || getShape(d) <= getPadding(2*d+1)) {
      output.clear() ;
      return getContext().setError
      (VLE_IllegalArgument,
       "PoolingForwardShape: an element of SHAPE is not larger than the corresponding PADDING.") ;
    }
    output.setDimension(d, odim) ;
  }
  output.setDimension(ns,input.getDimension(ns)) ;
  output.setDimension(ns+1,input.getDimension(ns+1)) ;
  return VLE_Success ;
}

vl::ErrorCode
Pooling::forward(Tensor &output,
                 Tensor const &input) const
{
  ErrorCode error ;

  // Validate arguments.
  if (!check_tensor_compatibility(output,input)) {
    return getContext().setError
    (VLE_IllegalArgument,
     "PoolingForward: the tensors have mismatching data or device type.") ;
  }
  TensorShape outputShape ;
  if ((error = forwardShape(outputShape, input)) != VLE_Success) {
    return error ;
  }
  if (output != outputShape) {
    return getContext().setError
    (VLE_TensorShapeMismatch,
     "PoolingForward: OUTPUT does not have the appropriate dimensions.") ;
  }
  if (input.isEmpty() | input.isNull()) {
    return getContext().setError
    (VLE_IllegalArgument,
     "PoolingForward: INPUT is emtpy or null.") ;
  }
  if (output.isEmpty() | output.isNull()) {
    return  getContext().setError
    (VLE_IllegalArgument,
     "PoolingForward: OUTPUT is empty or null.") ;
  }

  VLLOG(*this,1)
  << "Pooling: forward"
  << " shape=" << pretty(getShape())
  << " stride=" << pretty(getStrides())
  << " padding=" << pretty(getPaddings())
  << " method=" << ((getMethod() == Average) ? "average" : "max") ;

  VLLOG(*this,1)
  << "Pooling: input=" << pretty(input.getDimensions())
  << " output=" << pretty(output.getDimensions()) ;

  return getContext().passError
  (dispatch_cudnn<PoolingForward,PoolingForwardCudnn>()(*this,output,input),
   "PoolingForawd") ;
}

vl::ErrorCode
Pooling::backward(Tensor &derInput,
                  Tensor const &input,
                  Tensor const &derOutput) const
{
  // Validate arguments.
  vl::ErrorCode error ;
  if (!check_tensor_compatibility(derInput,input,derInput)) {
    return getContext().setError
    (VLE_IllegalArgument,
     "PoolingBackward: the tensors have mismatching data or device type.") ;
  }
  TensorShape outputShape ;
  if ((error = forwardShape(outputShape, input)) != VLE_Success) {
    return error ;
  }
  if (derOutput != outputShape) {
    return getContext().setError
    (VLE_TensorShapeMismatch,
     "PoolingBackward: DEROUTPUT does not have the appropriate dimensions.") ;
  }
  if (input.isEmpty() | (input.isNull() && getMethod() != Average)) {
    return getContext().setError
    (VLE_IllegalArgument,
     "PoolingBackward: INPUT is emtpy (or null and the pooling method is not Average).") ;
  }
  if (derInput.isEmpty() | derInput.isNull()) {
    return getContext().setError
    (VLE_IllegalArgument,
     "PoolingBackward: DERINPUT is empty or null.") ;
  }
  if (static_cast<TensorShape>(derInput) != static_cast<TensorShape>(input)) {
    return getContext().setError
    (VLE_IllegalArgument,
     "ConvolutionBackward: DERINPUT size is not the same as INPUT.") ;
  }

  VLLOG(*this,1)
  << "Pooling: backward"
  << " stride=" << pretty(getStrides())
  << " padding=" << pretty(getPaddings())
  << " method=" << ((getMethod() == Average) ? "average" : "max") ;

  VLLOG(*this,1)
  << "Pooling: derInput=" << pretty(derInput.getDimensions())
  << " input=" << pretty(input.getDimensions())
  << " derOutput=" << pretty(derOutput.getDimensions()) ;

  return getContext().passError
  (dispatch_cudnn<PoolingBackward,PoolingBackwardCudnn>()(*this,derInput,input,derOutput),
   "PoolingBackward") ;
}
