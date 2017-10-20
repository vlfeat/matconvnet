// @file nnpooling_gpu.cu
// @brief Pooling block GPU.
// @author Andrea Vedaldi
// @author Karel Lenc

/*
Copyright (C) 2014-17 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnpooling.hpp"
#include "datacu.hpp"
#include <cassert>
#include <float.h>
#include <sm_20_atomic_functions.h>

// -------------------------------------------------------------------
//                                                 Max pooling helpers
// -------------------------------------------------------------------

template<typename T> __global__ void
pooling_max_kernel
(T* output,
 const T* data,
 const int outputWidth,
 const int outputHeight,
 const int outputVolume,
 const int width,
 const int height,
 const int poolWidth,
 const int poolHeight,
 const int strideX,
 const int strideY,
 const int padLeft,
 const int padTop)
{
  int outputIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (outputIndex < outputVolume) {
    int px = outputIndex ;
    int py = px / outputWidth ;
    int pz = py / outputHeight ;
    px %= outputWidth ;
    py %= outputHeight ;
    data += pz * (width*height) ;

    int x1 = px * strideX - padLeft ;
    int y1 = py * strideY - padTop ;
    int x2 = min(x1 + poolWidth, width) ;
    int y2 = min(y1 + poolHeight, height) ;
    x1 = max(x1, 0) ;
    y1 = max(y1, 0) ;

    T bestValue = data[y1 * width + x1] ;
    for (int y = y1 ; y < y2 ; ++y) {
      for (int x = x1 ; x < x2 ; ++x) {
        bestValue = max(bestValue, data[y * width + x]) ;
      }
    }
    output[outputIndex] = bestValue ;
  }
}

#ifdef VLNN_CAFFELIKE_BPPOOL
// In order to be able to use this, BP would need to have access to both
// bottom data and output data (currently only passed bottom data...)
template <typename T> __global__ void
pooling_max_backward_with_output_data
(T* derData,
 const T* data,
 const T* output,
 const T* derOutput,
 const int nthreads,
 const int outputWidth,
 const int outputHeight,
 const int width,
 const int height,
 const int depth,
 const int poolWidth,
 const int poolHeight,
 const int strideX,
 const int strideY)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < nthreads) {
    // find out the local index
    // find out the local offset
    int x = index % width;
    int y = (index / width) % height;
    int z = (index / width / height) % depth;
    int py1 = (y < poolHeight) ? 0 : (y - poolHeight) / strideY + 1;
    int py2 = min(y / strideY + 1, outputHeight);
    int px1 = (x < poolWidth) ? 0 : (x - poolWidth) / strideX + 1;
    int px2 = min(x / strideX + 1, outputWidth);
    T gradient = 0;
    T datum = data[(z * height + y) * width + x];
    output += z * outputHeight * outputWidth;
    dzdy += z * outputHeight * outputWidth;
    for (int py = py1; py < py2; ++py) {
      for (int px = px1; px < px2; ++px) {
        gradient += dzdy[py * outputWidth + px] *
        (datum == output[py * outputWidth + px]);
      }
    }
    dzdx[index] = gradient;
  }
}
#endif

template<typename T> __global__ void
pooling_max_backward_kernel
(T* derData,
 const T* data,
 const T* derOutput,
 const int outputWidth,
 const int outputHeight,
 const int outputVolume,
 const int width,
 const int height,
 const int poolWidth,
 const int poolHeight,
 const int strideX,
 const int strideY,
 const int padLeft,
 const int padTop)
{
  int outputIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (outputIndex < outputVolume) {
    int px = outputIndex ;
    int py = px / outputWidth ;
    int pz = py / outputHeight ;
    px %= outputWidth ;
    py %= outputHeight ;
    data += pz * (width*height) ;
    derData += pz * (width*height) ;

    int x1 = px * strideX - padLeft ;
    int y1 = py * strideY - padTop ;
    int x2 = min(x1 + poolWidth, width) ;
    int y2 = min(y1 + poolHeight, height) ;
    x1 = max(x1, 0) ;
    y1 = max(y1, 0) ;

    int bestIndex = y1 * width + x1 ;
    T bestValue = data[bestIndex] ;
    for (int y = y1 ; y < y2 ; ++y) {
      for (int x = x1 ; x < x2 ; ++x) {
        int index = y * width + x ;
        T value = data[index] ;
        if (value > bestValue) {
          bestValue = value ;
          bestIndex = index ;
        }
      }
    }
    /*
     This is bad, but required to eliminate a race condition when writing
     to bottom_diff.
     Caffe goes the other way around, but requrires remembering the layer
     output, or the maximal indexes.
     atomicAdd(add, val)
     */
    atomicAdd(derData + bestIndex, derOutput[outputIndex]) ;
  }
}

// -------------------------------------------------------------------
//                                             Average pooling helpers
// -------------------------------------------------------------------

template<typename T> __global__ void
pooling_average_kernel
(T* output,
 const T* data,
 const int outputWidth,
 const int outputHeight,
 const int outputVolume,
 const int width,
 const int height,
 const int poolWidth,
 const int poolHeight,
 const int strideX,
 const int strideY,
 const int padLeft,
 const int padTop)
{
  /* outputIndex = x + y * outputWidth + z * (outputWidth * outputHeight) */
  int outputIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (outputIndex < outputVolume) {
    int px = outputIndex ;
    int py = px / outputWidth ;
    int pz = py / outputHeight ;
    px %= outputWidth ;
    py %= outputHeight ;
    int x1 = px * strideX - padLeft ;
    int y1 = py * strideY - padTop ;
    int x2 = min(x1 + poolWidth, width) ;
    int y2 = min(y1 + poolHeight, height) ;
    x1 = max(x1, 0) ;
    y1 = max(y1, 0) ;
    data += pz * (width*height) ;
    T accum = 0;
    T poolSize = (y2 - y1)*(x2 - x1);
    for (int y = y1 ; y < y2 ; ++y) {
      for (int x = x1 ; x < x2 ; ++x) {
        accum += data[y * width + x] ;
      }
    }
    output[outputIndex] = accum / poolSize ;
  }
}

template <typename T> __global__ void
pooling_average_backward_kernel
(T* derData,
 const T* derOutput,
 const int nthreads,
 const int outputWidth,
 const int outputHeight,
 const int width,
 const int height,
 const int depth,
 const int poolWidth,
 const int poolHeight,
 const int strideX,
 const int strideY,
 const int padLeft,
 const int padTop)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < nthreads) {
    /* To understand the logic of this piece of code see the
     comments to of the row2im backward kernel */
    int x_data = index ;
    int y_data = x_data / width ;
    int z = y_data / height ;
    x_data %= width ;
    y_data %= height ;

    int dx = x_data + padLeft - poolWidth ;
    int dy = y_data + padTop - poolHeight ;
    int px1 = (dx >= 0) ? dx/strideX + 1 : 0 ;
    int py1 = (dy >= 0) ? dy/strideY + 1 : 0 ;
    int px2 = min((x_data + padLeft) / strideX, outputWidth - 1) ;
    int py2 = min((y_data + padTop) / strideY, outputHeight - 1) ;
    T accumulator = 0 ;
    derOutput += z * outputHeight * outputWidth;
    for (int py = py1 ; py <= py2 ; ++py) {
      for (int px = px1 ; px <= px2 ; ++px) {
        int x1 = px * strideX - padLeft ;
        int y1 = py * strideY - padTop ;
        int x2 = min(x1 + poolWidth, width) ;
        int y2 = min(y1 + poolHeight, height) ;
        x1 = max(x1, 0) ;
        y1 = max(y1, 0) ;
        T poolSize = (y2 - y1) * (x2 - x1);
        accumulator += derOutput[py * outputWidth + px] / poolSize ;
      }
    }
    derData[index] = accumulator ;
  }
}

// -------------------------------------------------------------------
//                                                             Forward
// -------------------------------------------------------------------

template<DataType dataType, Pooling::Method method>
struct PoolingForwardGPU
{
  vl::ErrorCode operator()(Pooling const &op,
                           Tensor &output,
                           Tensor const &input)
  {
    // Argument sanity check.
    assert(output) ;
    assert(input) ;
    TensorShape outShape ;
    op.forwardShape(outShape, input) ;
    assert(outShape == output) ;

    typedef typename vl::DataTypeTraits<dataType>::type type ;
    Int outputVolume = output.getNumElements() ;

    if (method == Pooling::Max) {
      pooling_max_kernel<type>
      <<< divideAndRoundUp((unsigned)outputVolume,VL_CUDA_NUM_THREADS),VL_CUDA_NUM_THREADS >>>
      ((type*)output.getMemory(),
       (type const*)input.getMemory(),
       (int)output.getHeight(), (int)output.getWidth(), (int)output.getNumElements(),
       (int)input.getHeight(), (int)input.getWidth(),
       (int)op.getShape(0), (int)op.getShape(1),
       (int)op.getStride(0), (int)op.getStride(1),
       (int)op.getPadding(0), (int)op.getPadding(2));
    }
    else if (method == Pooling::Average) {
      pooling_average_kernel<type>
      <<< divideAndRoundUp((unsigned)outputVolume,VL_CUDA_NUM_THREADS),VL_CUDA_NUM_THREADS >>>
      ((type*)output.getMemory(),
       (type const*)input.getMemory(),
       (int)output.getHeight(), (int)output.getWidth(), (int)output.getNumElements(),
       (int)input.getHeight(), (int)input.getWidth(),
       (int)op.getShape(0), (int)op.getShape(1),
       (int)op.getStride(0), (int)op.getStride(1),
       (int)op.getPadding(0), (int)op.getPadding(2));
    }
    else {
      assert(false) ;
    }

    cudaError_t status = cudaPeekAtLastError() ;
    return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
  }
} ;

template<DataType dataType>
struct PoolingForward<VLDT_GPU,dataType>
{
  vl::ErrorCode operator()(Pooling const &op,
                           Tensor output,
                           Tensor input)
  {
    switch (op.getMethod()) {
      case Pooling::Max:
        return
        PoolingForwardGPU<dataType,Pooling::Max>
        ()(op,output,input) ;
      case Pooling::Average:
        return
        PoolingForwardGPU<dataType,Pooling::Average>
        ()(op,output,input) ;
      default:
        return VLE_IllegalArgument ;
    }
  }
} ;

// -------------------------------------------------------------------
//                                                            Backward
// -------------------------------------------------------------------

template<DataType dataType, Pooling::Method method>
struct PoolingBackwardGPU
{
  vl::ErrorCode operator()(Pooling const &op,
                           Tensor &derInput,
                           Tensor const &input,
                           Tensor const &derOutput)
  {
    // Argument sanity check.
    assert(derInput) ;
    assert(derOutput) ;
    TensorShape outputShape ;
    op.forwardShape(outputShape,derInput) ;
    assert(derOutput == outputShape) ;

    typedef typename vl::DataTypeTraits<dataType>::type type ;
    Int outputVolume = derOutput.getNumElements() ;
    Int inputVolume = input.getNumElements() ;

    if (method == Pooling::Max) {
      assert(input) ;
      pooling_max_backward_kernel<type>
      <<< divideAndRoundUp((unsigned)outputVolume,VL_CUDA_NUM_THREADS),VL_CUDA_NUM_THREADS >>>
      ((type*)derInput.getMemory(),
       (type const*)input.getMemory(),
       (type const*)derOutput.getMemory(),
       (int)derOutput.getHeight(), (int)derOutput.getWidth(), (int)outputVolume,
       (int)input.getHeight(), (int)input.getWidth(),
       (int)op.getShape(0), (int)op.getShape(1),
       (int)op.getStride(0), (int)op.getStride(1),
       (int)op.getPadding(0), (int)op.getPadding(2));
    }
    else if (method == Pooling::Average) {
      pooling_average_backward_kernel<type>
      <<< divideAndRoundUp((unsigned)inputVolume,VL_CUDA_NUM_THREADS),VL_CUDA_NUM_THREADS >>>
      ((type*)derInput.getMemory(),
       (type const*)derOutput.getMemory(),
       (int)inputVolume,
       (int)derOutput.getHeight(), (int)derOutput.getWidth(),
       (int)input.getHeight(), (int)input.getWidth(), (int)(input.getDepth()*input.getSize()),
       (int)op.getShape(0), (int)op.getShape(1),
       (int)op.getStride(0), (int)op.getStride(1),
       (int)op.getPadding(0), (int)op.getPadding(2)) ;
    }
    else {
      assert(false) ;
    }

    cudaError_t status = cudaPeekAtLastError() ;
    return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
  }
} ; // pooling_max


template<DataType dataType>
struct PoolingBackward<VLDT_GPU,dataType>
{
  vl::ErrorCode operator()(Pooling const &op,
                           Tensor &derInput,
                           Tensor const &input,
                           Tensor const &derOutput)
  {
    switch (op.getMethod()) {
      case Pooling::Max:
        return
        PoolingBackwardGPU<dataType,Pooling::Max>
        ()(op,derInput,input,derOutput) ;
      case Pooling::Average:
        return
        PoolingBackwardGPU<dataType,Pooling::Average>
        ()(op,derInput,input,derOutput) ;
      default:
        return VLE_IllegalArgument ;
    }
  }
} ;
