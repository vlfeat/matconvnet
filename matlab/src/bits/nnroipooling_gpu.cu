// @file nnroipooling_gpu.cu
// @brief ROI pooling block (GPU)
// @author Hakan Bilen
// @author Abishek Dutta
// @author Andrea Vedaldi

/*
Copyright (C) 2016-17 Hakan Bilen, Abishek Dutta, and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "datacu.hpp"

#include <assert.h>
#include <cfloat>
#include <algorithm>
#include <sm_20_atomic_functions.h>

// -------------------------------------------------------------------
//                                                             Helpers
// -------------------------------------------------------------------

template<typename T>
struct Geom {
  int subdivisions[2] ;
  T transform[6] ;
  Geom(std::array<int,2> const &subdivisions, std::array<double,6> const &transform)
  {
    this->subdivisions[0] = subdivisions[0] ;
    this->subdivisions[1] = subdivisions[1] ;
    this->transform[0] = transform[0] ;
    this->transform[1] = transform[1] ;
    this->transform[2] = transform[2] ;
    this->transform[3] = transform[3] ;
    this->transform[4] = transform[4] ;
    this->transform[5] = transform[5] ;
  }
} ;

struct Bounds {
  int image, offset, hstart, hend, wstart, wend ;
  bool isEmpty ;
} ;

template<typename T>
__device__ __forceinline__ static Bounds
getBounds(int outputIndex,
          int height, int width, int numChannels, int size,
          const T* rois, int numROIs,
          Geom<T> geom)
{
  Bounds b ;

  int ph = outputIndex ;
  int pw = ph / geom.subdivisions[0] ;
  int pc = pw / geom.subdivisions[1] ;
  int pr = pc / numChannels ;

  ph %= geom.subdivisions[0] ;
  pw %= geom.subdivisions[1] ;
  pc %= numChannels ;

  rois += 5 * pr ;

  // Apply sacle and offset to each ROI coordinate.
  T u1_ = rois[1] ;
  T v1_ = rois[2] ;
  T u2_ = rois[3] ;
  T v2_ = rois[4] ;

  T u1 = geom.transform[0] * u1_ + geom.transform[2] * v1_ + geom.transform[4] ;
  T v1 = geom.transform[1] * u1_ + geom.transform[3] * v1_ + geom.transform[5] ;
  T u2 = geom.transform[0] * u2_ + geom.transform[2] * v2_ + geom.transform[4] ;
  T v2 = geom.transform[1] * u2_ + geom.transform[3] * v2_ + geom.transform[5] ;

  // First and last pixel of each ROI (rounded
  // for compatibility with the Caffe definition).
  int roi_image   = (int)rois[0];
  int roi_start_h = (int)::round(v1) - 1 ;
  int roi_start_w = (int)::round(u1) - 1 ;
  int roi_end_h   = (int)::round(v2) - 1 ;
  int roi_end_w   = (int)::round(u2) - 1 ;
  int roi_height  = max(roi_end_h - roi_start_h + 1, 1) ;
  int roi_width   = max(roi_end_w - roi_start_w + 1, 1) ;

  T bin_size_h = (T)roi_height / geom.subdivisions[0] ;
  T bin_size_w = (T)roi_width / geom.subdivisions[1] ;

  roi_image = min(max(roi_image - 1,0), (int)size - 1) ;
  b.offset = (roi_image * numChannels + pc) * (width*height) ;

  b.wstart = (int)floor(((T)pw) * bin_size_w) ;
  b.wend = (int)ceil(((T)(pw + 1)) * bin_size_w) ;
  b.wstart = min(max(b.wstart + roi_start_w, 0), (int)width) ;
  b.wend = min(max(b.wend + roi_start_w, 0), (int)width) ;

  b.hstart = (int)floor(((T)ph) * bin_size_h) ;
  b.hend = (int)ceil(((T)(ph + 1)) * bin_size_h) ;
  b.hstart = min(max(b.hstart + roi_start_h, 0), (int)height) ;
  b.hend = min(max(b.hend + roi_start_h, 0), (int)height) ;

  b.isEmpty = (b.hend <= b.hstart) || (b.wend <= b.wstart) ;

  return b ;
}

/* ---------------------------------------------------------------- */
/*                                       roipooling_average_forward */
/* ---------------------------------------------------------------- */

template<typename T> __global__ void
roipooling_average_kernel
(T* output,
 const T* data, int height, int width, int numChannels, int size,
 const T* rois, int numROIs,
 Geom<T> geom)
{
  int outputIndex = threadIdx.x + blockIdx.x * blockDim.x;
  int outputVolume = geom.subdivisions[0] * geom.subdivisions[1] * numChannels * numROIs;
  if (outputIndex < outputVolume) {
    Bounds b = getBounds<T>(outputIndex,
                            height,width,numChannels,size,
                            rois,numROIs,
                            geom) ;
    data += b.offset ;
    T bestValue = 0;
    const T coeff = ((T)1.) / (T)((b.wend-b.wstart) * (b.hend-b.hstart));
    for (int w = b.wstart; w < b.wend; ++w) {
      for (int h = b.hstart; h < b.hend; ++h) {
        int index = w * height + h ;
        bestValue += data[index] * coeff ;
      }
    }
    output[outputIndex] = bestValue ;
  }
}

/* ---------------------------------------------------------------- */
/*                                           roipooling_max_forward */
/* ---------------------------------------------------------------- */

template<typename T> __global__ void
roipooling_max_kernel
(T* output,
 const T* data, int height, int width, int numChannels, int size,
 const T* rois, int numROIs,
 Geom<T> geom)
{
  int outputIndex = threadIdx.x + blockIdx.x * blockDim.x ;
  int outputVolume = geom.subdivisions[0] * geom.subdivisions[1] * numChannels * numROIs ;
  if (outputIndex < outputVolume) {
    Bounds b = getBounds<T>(outputIndex,
                            height,width,numChannels,size,
                            rois,numROIs,
                            geom) ;
    data += b.offset ;
    if (! b.isEmpty) {
      T bestValue = -FLT_MAX;
      for (int w = b.wstart; w < b.wend; ++w) {
        for (int h = b.hstart; h < b.hend; ++h) {
          int index = w * height + h ;
          bestValue = max(bestValue, data[index]) ;
        }
      }
      output[outputIndex] = bestValue ;
    } else {
      output[outputIndex] = 0 ;
    }
  }
}

/* ---------------------------------------------------------------- */
/*                                      roipooling_average_backward */
/* ---------------------------------------------------------------- */

template<typename T> __global__ void
roipooling_average_backward_kernel
(T* derData,
 const T* data, int height, int width, int numChannels, int size,
 const T* rois, int numROIs,
 const T* derOutput,
 Geom<T> geom)
{
  int outputIndex = threadIdx.x + blockIdx.x * blockDim.x;
  int outputVolume = geom.subdivisions[0] * geom.subdivisions[1] * numChannels * numROIs ;
  if (outputIndex < outputVolume) {

    Bounds b = getBounds<T>(outputIndex,
                            height,width,numChannels,size,
                            rois,numROIs,
                            geom) ;
    data += b.offset ;
    derData += b.offset ;
    const T coeff = ((T)1.) / (T)((b.wend-b.wstart)*(b.hend-b.hstart)) ;
    for (int h = b.hstart; h < b.hend; ++h) {
      for (int w = b.wstart; w < b.wend; ++w) {
        int index = w * height + h ;
        atomicAdd(derData + index, derOutput[outputIndex] * coeff) ;
      }
    }
  }
}

/* ---------------------------------------------------------------- */
/*                                          roipooling_max_backward */
/* ---------------------------------------------------------------- */

template<typename T> __global__ void
roipooling_max_backward_kernel
(T* derData,
 const T* data, int height, int width, int numChannels, int size,
 const T* rois, int numROIs,
 const T* derOutput,
 Geom<T> geom)
{
  int outputIndex = threadIdx.x + blockIdx.x * blockDim.x;
  int outputVolume = geom.subdivisions[0] * geom.subdivisions[1] * numChannels * numROIs;

  if (outputIndex < outputVolume) {

    Bounds b = getBounds<T>(outputIndex,
                            height,width,numChannels,size,
                            rois,numROIs,
                            geom) ;
    if (! b.isEmpty) {
      data += b.offset ;
      derData += b.offset ;
      int bestIndex = min(b.wstart,width-1) * height + min(b.hstart,height-1);
      T bestValue = -FLT_MAX;
      for (int h = b.hstart; h < b.hend; ++h) {
        for (int w = b.wstart; w < b.wend; ++w) {
          int index = w * height + h ;
          T value = data[index] ;
          if (value > bestValue) {
            bestValue = value ;
            bestIndex = index ;
          }
        }
      }
      atomicAdd(derData + bestIndex, derOutput[outputIndex]) ;
    }
  }
}

// -------------------------------------------------------------------
//                                                             Forward
// -------------------------------------------------------------------

template<DataType dataType, ROIPooling::Method method>
struct ROIPoolingForwardGPU
{
  vl::ErrorCode operator()(ROIPooling &op,
                           Tensor &output,
                           Tensor const &input,
                           Tensor const &rois)
  {
    typedef typename vl::DataTypeTraits<dataType>::type type ;
    auto numROIs = rois.getNumElements() / 5 ;
    auto outputData = (type*)output.getMemory() ;
    auto inputData = (type const*)input.getMemory() ;
    auto height = input.getHeight() ;
    auto width = input.getWidth() ;
    auto numChannels = input.getDepth() ;
    auto size = input.getSize() ;
    auto roisData = (type const*)rois.getMemory() ;
    size_t outputVolume = op.subdivisions[0] * op.subdivisions[1] * numChannels * numROIs ;

    if (method == ROIPooling::Max) {
      roipooling_max_kernel<type>
      <<< divideAndRoundUp(outputVolume, (size_t)VL_CUDA_NUM_THREADS),VL_CUDA_NUM_THREADS >>>
      (outputData,
       inputData, height, width, numChannels, size,
       roisData, numROIs,
       Geom<type>(op.subdivisions,op.transform)) ;
    }
    else if (method == ROIPooling::Average) {
      roipooling_average_kernel<type>
      <<< divideAndRoundUp(outputVolume, (size_t)VL_CUDA_NUM_THREADS),VL_CUDA_NUM_THREADS >>>
      (outputData,
       inputData, height, width, numChannels, size,
       roisData, numROIs,
       Geom<type>(op.subdivisions,op.transform)) ;
    }
    else {
      assert(false) ;
    }

    cudaError_t status = cudaPeekAtLastError() ;
    return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
  }
} ;

template<DataType dataType>
struct ROIPoolingForward<VLDT_GPU,dataType>
{
  vl::ErrorCode operator()(ROIPooling &op,
                           Tensor pooled,
                           Tensor input,
                           Tensor rois)
  {
    switch (op.method) {
      case ROIPooling::Max:
        return ROIPoolingForwardGPU<dataType,ROIPooling::Max>
        ()(op,pooled,input,rois) ;
      case ROIPooling::Average:
        return ROIPoolingForwardGPU<dataType,ROIPooling::Average>
        ()(op,pooled,input,rois) ;
      default: return VLE_IllegalArgument ;
    }
  }
} ;

// -------------------------------------------------------------------
//                                                            Backward
// -------------------------------------------------------------------

template<DataType dataType, ROIPooling::Method method>
struct ROIPoolingBackwardGPU
{
  vl::ErrorCode operator()(ROIPooling &op,
                           Tensor &derInput,
                           Tensor const &input,
                           Tensor const &rois,
                           Tensor const &derOutput)
  {
    typedef typename vl::DataTypeTraits<dataType>::type type ;
    auto numROIs = rois.getNumElements() / 5 ;
    auto derInputData = (type*)derInput.getMemory() ;
    auto derOutputData = (type const*)derOutput.getMemory() ;
    auto inputData = (type const*)input.getMemory() ;
    auto height = input.getHeight() ;
    auto width = input.getWidth() ;
    auto numChannels = input.getDepth() ;
    auto size = input.getSize() ;
    auto roisData = (type const*)rois.getMemory() ;
    size_t outputVolume = op.subdivisions[0] * op.subdivisions[1] * numChannels * numROIs ;

    if (method == ROIPooling::Max) {
      roipooling_max_backward_kernel<type>
      <<< divideAndRoundUp(outputVolume, (size_t)VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (derInputData, inputData,
       height, width, numChannels, size,
       roisData, numROIs,
       derOutputData,
       Geom<type>(op.subdivisions,op.transform)) ;
    }
    else if (method == ROIPooling::Average) {
      roipooling_average_backward_kernel<type>
      <<< divideAndRoundUp(outputVolume, (size_t)VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (derInputData, inputData,
       height, width, numChannels, size,
       roisData, numROIs,
       derOutputData,
       Geom<type>(op.subdivisions,op.transform)) ;
    }
    else {
      assert(false) ;
    }
    cudaError_t status = cudaPeekAtLastError() ;
    return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
  }
} ;

template<DataType dataType>
struct ROIPoolingBackward<VLDT_GPU,dataType>
{
  vl::ErrorCode operator()(ROIPooling &op,
                           Tensor &derInput,
                           Tensor const &input,
                           Tensor const &rois,
                           Tensor const &derOutput)
  {
    switch (op.method) {
      case ROIPooling::Max:
        return ROIPoolingBackwardGPU<dataType,ROIPooling::Max>
        ()(op,derInput,input,rois,derOutput) ;
      case ROIPooling::Average:
        return ROIPoolingBackwardGPU<dataType,ROIPooling::Average>
        ()(op,derInput,input,rois,derOutput) ;
      default: return VLE_IllegalArgument ;
    }
  }
} ;

