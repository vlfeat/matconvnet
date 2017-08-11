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
  inline acc_max(int poolHeight, int poolWidth, type derOutput = 0)
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
  inline acc_sum(int poolHeight, int poolWidth, type derOutput = 0)
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
  vl::ErrorCode operator()(Pooling &op,
                           Tensor &output,
                           Tensor const &input)
  {
    typedef typename vl::DataTypeTraits<dataType>::type type ;
    auto height = input.getHeight() ;
    auto width = input.getWidth() ;
    auto depth = input.getDepth() ;
    auto size = input.getSize() ;
    auto inputData = (type const*)input.getMemory() ;
    auto outputData = (type*)output.getMemory() ;
    auto outputWidth = (width + (op.padLeft + op.padRight) - op.poolWidth)/op.strideX + 1 ;
    auto outputHeight = (height + (op.padTop + op.padBottom) - op.poolHeight)/op.strideY + 1 ;

    for (int z = 0; z < depth * size ; ++z) {
      for (int x = 0; x < outputWidth; ++x) {
        for (int y = 0; y < outputHeight; ++y) {
          int x1 = x * (signed)op.strideX - (signed)op.padLeft ;
          int y1 = y * (signed)op.strideY - (signed)op.padTop ;
          int x2 = std::min(x1 + op.poolWidth, (int)width) ;
          int y2 = std::min(y1 + op.poolHeight, (int)height) ;
          x1 = std::max(x1, 0) ;
          y1 = std::max(y1, 0) ;
          Accumulator acc(y2 - y1, x2 - x1) ;
          for (int u = x1 ; u < x2 ; ++u) {
            for (int v = y1 ; v < y2 ; ++v) {
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
  vl::ErrorCode operator()(Pooling &op,
                           Tensor output,
                           Tensor input)
  {
    switch (op.method) {
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
  vl::ErrorCode operator()(Pooling &op,
                           Tensor &derInput,
                           Tensor const &input,
                           Tensor const &derOutput)
  {
    typedef typename vl::DataTypeTraits<dataType>::type type ;
    auto height = input.getHeight() ;
    auto width = input.getWidth() ;
    auto depth = input.getDepth() ;
    auto size = input.getSize() ;
    auto derInputData = (type*)derInput.getMemory() ;
    auto inputData = (type const*)input.getMemory() ;
    auto derOutputData = (type const*)derOutput.getMemory() ;
    auto outputWidth = (width + (op.padLeft + op.padRight) - op.poolWidth)/op.strideX + 1 ;
    auto outputHeight = (height + (op.padTop + op.padBottom) - op.poolHeight)/op.strideY + 1 ;

    for (int z = 0; z < depth * size ; ++z) {
      for (int x = 0; x < outputWidth ; ++x) {
        for (int y = 0; y < outputHeight; ++y) {
          int x1 = x * (signed)op.strideX - (signed)op.padLeft ;
          int y1 = y * (signed)op.strideY - (signed)op.padTop ;
          int x2 = std::min(x1 + op.poolWidth, (int)width) ;
          int y2 = std::min(y1 + op.poolHeight, (int)height) ;
          x1 = std::max(x1, 0) ;
          y1 = std::max(y1, 0) ;
          Accumulator acc(y2 - y1, x2 - x1, derOutputData[x * outputHeight + y]) ;
          for (int u = x1 ; u < x2 ; ++u) {
            for (int v = y1 ; v < y2 ; ++v) {
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
  vl::ErrorCode operator()(Pooling &op,
                           Tensor &derInput,
                           Tensor const &input,
                           Tensor const &derOutput)
  {
    switch (op.method) {
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
                 int poolHeight, int poolWidth,
                 int strideY, int strideX,
                 int padTop, int padBottom,
                 int padLeft, int padRight,
                 Method method) :
context(context),
poolHeight(poolHeight),
poolWidth(poolWidth),
strideY(strideY),
strideX(strideX),
padTop(padTop),
padBottom(padBottom),
padLeft(padLeft),
padRight(padRight),
method(method)
{ }

vl::ErrorCode
Pooling::forward(Tensor &output,
                 Tensor const &input)
{
  return dispatch_cudnn<PoolingForward,PoolingForwardCudnn>()(*this,output,input) ;
}

vl::ErrorCode
Pooling::backward(Tensor &derInput,
                  Tensor const &input,
                  Tensor const &derOutput)
{
  return dispatch_cudnn<PoolingBackward,PoolingBackwardCudnn>()(*this,derInput,input,derOutput) ;
}
