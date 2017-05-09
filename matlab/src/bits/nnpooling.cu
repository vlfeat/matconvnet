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

using namespace std ;
using namespace vl ;
using namespace vl::nn ;

template<DeviceType deviceType, DataType dataType> struct PoolingMaxForward ;
template<DeviceType deviceType, DataType dataType> struct PoolingMaxBackward ;
template<DeviceType deviceType, DataType dataType> struct PoolingAverageForward ;
template<DeviceType deviceType, DataType dataType> struct PoolingAverageBackward ;

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
struct PoolingForward
{
  vl::ErrorCode operator()(Pooling &op,
                           Tensor output,
                           Tensor input)
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
          int x2 = min(x1 + op.poolWidth, (int)width) ;
          int y2 = min(y1 + op.poolHeight, (int)height) ;
          x1 = max(x1, 0) ;
          y1 = max(y1, 0) ;
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
struct PoolingMaxForward<VLDT_CPU,dataType> :
public PoolingForward<dataType,acc_max<typename vl::DataTypeTraits<dataType>::type> >
{ } ;

template<DataType dataType>
struct PoolingAverageForward<VLDT_CPU,dataType> :
public PoolingForward<dataType,acc_sum<typename vl::DataTypeTraits<dataType>::type> >
{ } ;

// -------------------------------------------------------------------
//                                                            Backward
// -------------------------------------------------------------------

template<DataType dataType, class Accumulator>
struct PoolingBackward
{
  vl::ErrorCode operator()(Pooling &op,
                           Tensor derInput,
                           Tensor input,
                           Tensor derOutput)
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
      for (int x = 0; x < outputWidth; ++x) {
        for (int y = 0; y < outputHeight; ++y) {
          int x1 = x * (signed)op.strideX - (signed)op.padLeft ;
          int y1 = y * (signed)op.strideY - (signed)op.padTop ;
          int x2 = min(x1 + op.poolWidth, (int)width) ;
          int y2 = min(y1 + op.poolHeight, (int)height) ;
          x1 = max(x1, 0) ;
          y1 = max(y1, 0) ;
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
struct PoolingMaxBackward<VLDT_CPU,dataType> :
public PoolingBackward<dataType,acc_max<typename vl::DataTypeTraits<dataType>::type> >
{ } ;

template<DataType dataType>
struct PoolingAverageBackward<VLDT_CPU,dataType> :
public PoolingBackward<dataType,acc_sum<typename vl::DataTypeTraits<dataType>::type> >
{ } ;

// -------------------------------------------------------------------
//                                                              Driver
// -------------------------------------------------------------------

#if ENABLE_GPU
#include "nnPooling_gpu.cu"
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
Pooling::forward(vl::Tensor output,vl::Tensor input)
{
  switch (method) {
    case Max: return dispatch<PoolingMaxForward>()(*this,output,input) ;
    case Average: return dispatch<PoolingAverageForward>()(*this,output,input) ;
    default: return VLE_IllegalArgument ;
  }
}

vl::ErrorCode
Pooling::backward(vl::Tensor derInput,
                  vl::Tensor input,
                  vl::Tensor derOutput)
{
  switch (method) {
    case Max: return dispatch<PoolingMaxBackward>()(*this,derInput,input,derOutput) ;
    case Average: return dispatch<PoolingAverageBackward>()(*this,derInput,input,derOutput) ;
    default: return VLE_IllegalArgument ;
  }
}
