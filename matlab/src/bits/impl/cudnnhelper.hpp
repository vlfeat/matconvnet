// @file blashelper.hpp
// @brief BLAS helpers
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef cudnnhelper_h
#define cudnnhelper_h

#include "cudnn.h"
#include "assert.h"

#define COMMA ,

#if (CUDNN_VERSION >= 5000)
#define IF_CUDNN_GE5(x) x
#else
#define IF_CUDNN_GE5(x)
#endif

#if (CUDNN_VERSION >= 4000)
#define IF_CUDNN_GE4(x) x
#else
#define IF_CUDNN_GE4(x)
#endif

#if (CUDNN_VERSION >= 3000)
#define IF_CUDNN_GE3(x) x
#else
#define IF_CUDNN_GE3(x)
#endif

#if (CUDNN_VERSION >= 3000 & CUDNN_VERSION < 4000)
#define IF_CUDNN_GE3_LT4(x) x
#else
#define IF_CUDNN_GE3_LT4(x)
#endif


namespace vl { namespace impl {

  template <vl::Type dataType> struct DataTypeToCudnn { } ;
  template <> struct DataTypeToCudnn<vl::vlTypeFloat> { static cudnnDataType_t const id = CUDNN_DATA_FLOAT ; } ;
  template <> struct DataTypeToCudnn<vl::vlTypeDouble> { static cudnnDataType_t const id = CUDNN_DATA_DOUBLE ; } ;

  inline cudnnDataType_t dataTypeToCudnn(vl::Type dataType)
  {
    switch (dataType) {
      case vlTypeFloat: return DataTypeToCudnn<vl::vlTypeFloat>::id ;
      case vlTypeDouble: return DataTypeToCudnn<vl::vlTypeDouble>::id ;
      default: assert(false) ; return CUDNN_DATA_FLOAT ; // bogus
    }
  }

//  vl::Error createCudnnDescriptorFromTensor(Context & context,
//                                            cudnnTensorDescriptor_t * desriptor,
//                                            Tensor & const tensor)
//  {
//    cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
//    cudnnDataType_t cudnnDataType = dataTypeToCudnn(tensor.getDataType()) ;
//
//    cudnnError = cudnnCreateTensorDescriptor(descriptor) ;
//    if (cudnnError != CUDNN_STATUS_SUCCESS) {
//      return context.setError
//      (context.getCudaHelper().catchCudnnError(cudnnError, __func__)) ;
//    }
//
//    if (tensor.getNumDimensions() <= 4) {
//      size_t size = tensor.getSize() ;
//      size_t depth = tensor.getDepth() ;
//      size_t width = tensor.getWidth() ;
//      size_t height = tensor.getHeight() ;
//      cudnnError = cudnnSetTensor4dDescriptorExt(descriptor,
//                                                 CUDNN_TENSOR_NCHW,
//                                                 cudnnDataType,
//                                                 size, depth, width, height,
//                                                 depth * width * height, // strides
//                                                 width * height,
//                                                 height,
//                                                 1) ;
//    } else {
//      // todo: unimplemented
//      assert(false) ;
//    }
//    if (cudnnError != CUDNN_STATUS_SUCCESS) {
//      cudnnDestroyTensorDescriptor(descriptor) ;
//      return context.setError
//      (context.getCudaHelper().catchCudnnError(cudnnError, __func__)) ;
//    }
//    return vl::vlSuccess ;
//  }

} }



#endif /* cudnnhelper_h */
