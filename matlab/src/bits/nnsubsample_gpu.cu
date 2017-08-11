// @file nnsubsample_gpu.cu
// @brief Subsampling block implementation (GPU)
// @author Andrea Vedaldi
// @author Karel Lenc

/*
Copyright (C) 2014-16 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnsubsample.hpp"
#include "datacu.hpp"
#include <assert.h>
#include <float.h>
#include <iostream>

#ifndef ENABLE_GPU
#error "subsample_gpu.cu cannot be compiled without GPU support"
#endif

using namespace vl ;

// -------------------------------------------------------------------
//                                                             Helpers
// -------------------------------------------------------------------

template<typename T> __global__ void
subsample_forward_kernel
(T* output,
 const T* data,
 const int outputHeight,
 const int outputWidth,
 const int outputVolume,
 const int height,
 const int width,
 const int strideY,
 const int strideX,
 const int padTop,
 const int padLeft)
{
  int outputIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (outputIndex < outputVolume) {
    /* outputIndex = x
     + y * outputWidth
     + z * (outputWidth * outputHeight) ;
     */
    int py = outputIndex ;
    int px = py / outputHeight ;
    int channel = px / outputWidth ;
    px %= outputWidth ;
    py %= outputHeight ;
    int x1 = px * strideX - padLeft ;
    int y1 = py * strideY - padTop ;
    data += channel * (width*height) ;
    T value = 0 ;
    if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
      value = data[x1 * height + y1] ;
    }
    output[outputIndex] =  value ;
  }
}

template<typename T>
__global__ void subsample_backward_kernel
(T* derData,
 const T* derOutput,
 const int outputHeight,
 const int outputWidth,
 const int dataVolume,
 const int height,
 const int width,
 const int strideY,
 const int strideX,
 const int padTop,
 const int padLeft)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < dataVolume) {
    int y = index ;
    int x = y / height ;
    int channel = x / width ;
    x %= width ;
    y %= height ;
    derOutput += channel * outputHeight * outputWidth ;
    int px = (x + padLeft) / strideX ;
    int py = (y + padTop) / strideY ;
    if (x == strideX * px - padLeft &&
        y == strideY * py - padTop) {
      derData[index] = derOutput[px * outputHeight + py] ;
    } else {
      derData[index] = 0 ;
    }
  }
}
// -------------------------------------------------------------------
//                                                              Driver
// -------------------------------------------------------------------

template<vl::DataType dataType>
struct SubsampleForward<vl::VLDT_GPU, dataType>
{
  vl::ErrorCode operator()(Subsample &op,
                           Tensor &output,
                           Tensor const &input)
  {
    assert(output) ;
    assert(input) ;

    typedef typename vl::DataTypeTraits<dataType>::type type ;
    auto width = input.getWidth() ;
    auto height = input.getHeight() ;
    auto depth = input.getDepth() ;
    auto size = input.getSize() ;
    auto inputData = (type*)input.getMemory() ;
    auto outputData = (type*)output.getMemory() ;
    auto outputWidth = (width + (op.padLeft + op.padRight) - 1)/op.strideX + 1 ;
    auto outputHeight = (height + (op.padTop + op.padBottom) - 1)/op.strideY + 1 ;
    auto outputVolume = outputWidth * outputHeight * depth * size ;

    assert(outputWidth == output.getWidth()) ;
    assert(outputHeight == output.getHeight()) ;

    subsample_forward_kernel<type>
    <<< divideAndRoundUp(outputVolume, (size_t)VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
    (outputData, inputData,
     outputHeight, outputWidth, outputVolume,
     height, width,
     op.strideY, op.strideX,
     op.padTop, op.padLeft);
    return op.context.setError(op.context.getCudaHelper().catchCudaError(__func__)) ;
  }
} ;


template<vl::DataType dataType>
struct SubsampleBackward<vl::VLDT_GPU, dataType>
{
  vl::ErrorCode operator()(Subsample &op,
                           Tensor &derInput,
                           Tensor const &derOutput)
  {
    assert(derInput) ;
    assert(derOutput) ;

    typedef typename vl::DataTypeTraits<dataType>::type type ;
    auto width = derInput.getWidth() ;
    auto height = derInput.getHeight() ;
    auto depth = derInput.getDepth() ;
    auto size = derInput.getSize() ;
    auto volume = width * height * depth * size ;
    auto derInputData = (type*)derInput.getMemory() ;
    auto derOutputData = (type*)derOutput.getMemory() ;
    auto outputWidth = (width + (op.padLeft + op.padRight) - 1)/op.strideX + 1 ;
    auto outputHeight = (height + (op.padTop + op.padBottom) - 1)/op.strideY + 1 ;

    assert(outputWidth == derOutput.getWidth()) ;
    assert(outputHeight == derOutput.getHeight()) ;

    subsample_backward_kernel<type>
    <<< divideAndRoundUp(volume, (size_t)VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
    (derInputData,
     derOutputData,
     outputHeight, outputWidth, volume,
     height, width,
     op.strideY, op.strideX,
     op.padTop, op.padLeft);
    return op.context.setError(op.context.getCudaHelper().catchCudaError(__func__)) ;
  }
} ;
