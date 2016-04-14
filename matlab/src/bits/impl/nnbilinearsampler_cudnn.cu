// @file nnbilinearsampler_cudnn.cu
// @brief BilinearSampler CuDNN-based implementation.
// @author Ankush Gupta, Andrea Vedaldi

/*
Copyright (C) 2016- Ankush Gupta, Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#if !defined(ENABLE_GPU) | !defined(ENABLE_CUDNN)
#error "bilinearsampler_cudnn.hpp cannot be compiled without GPU and CUDNN support."
#endif

#include "nnbilinearsampler_cudnn.hpp"
#include "cudnnhelper.hpp"
#include "../datacu.hpp"
#include <assert.h>
#include <algorithm>

using namespace vl ;

// check if the descriptors, etc. were successfully created:
#define CHECK(x) \
{ \
cudnnError = x ; \
if (cudnnError != CUDNN_STATUS_SUCCESS) { \
error = context.setError(context.getCudaHelper().catchCudnnError(cudnnError, \
STRINGIZE(__FILE__) ":" STRINGIZE(__LINE__))) ; \
goto done ; \
} }

/* ---------------------------------------------------------------- */
/*                                    bilinearsampler_forward_cudnn */
/* ---------------------------------------------------------------- */
namespace vl { namespace impl {

  template<vl::Type dataType>
  vl::Error
  vl::impl::nnbilinearsampler_cudnn<dataType>::forward( Context& context,
                                                        Tensor output,
                                                        Tensor data,
                                                        Tensor grid )
  {
    assert(output) ;
    assert(data) ;
    assert(grid) ;

    typedef typename DataTypeTraits<dataType>::type type ;

    cudnnTensorDescriptor_t outputDesc, dataDesc ;
    cudnnSpatialTransformerDescriptor_t samplerDesc ;
    bool outputDescInitialized = false ;
    bool dataDescInitialized = false ;
    bool samplerDescInitialized = false ;

    // get the sizes:
    int inCardinality = data.getSize();
    int inDepth = data.getDepth();
    int inHeight = data.getHeight();
    int inWidth = data.getWidth();

    int outCardinality = output.getSize();
    int outDepth = output.getDepth();
    int outWidth = output.getWidth();
    int outHeight = output.getHeight();

    int dimOut[4] = { outCardinality, outDepth, outWidth, outHeight };
    
    cudnnDataType_t cudnnDataType = DataTypeToCudnn<dataType>::id ;
    vl::Type dynDataType = output.getDataType() ;
    assert(dynDataType == dataType) ;

    cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
    vl::Error error = vl::vlSuccess ;
    cudnnHandle_t handle ;

    // Get CuDNN
    CHECK(context.getCudaHelper().getCudnnHandle(&handle)) ;

    // Get tensor descriptors:
    CHECK(cudnnCreateTensorDescriptor(&outputDesc)) ;
    outputDescInitialized = true ;
  
    CHECK(cudnnSetTensor4dDescriptorEx(outputDesc,
                                       cudnnDataType,
                                       outCardinality, outDepth, outWidth, outHeight, // sizes: n,c,w,h
                                       outHeight * outWidth * outDepth, //strides
                                       outHeight * outWidth,
                                       outHeight,
                                       1)) ;

    CHECK(cudnnCreateTensorDescriptor(&dataDesc)) ;
    dataDescInitialized = true ;
  
    CHECK(cudnnSetTensor4dDescriptorEx(dataDesc,
                                       cudnnDataType,
                                       inCardinality, inDepth, inWidth, inHeight, // sizes: n,c,w,h
                                       inHeight * inWidth * inDepth, //strides
                                       inHeight * inWidth,
                                       inHeight,
                                       1)) ;

    // Get bilinear-sampler descriptor:
    CHECK(cudnnCreateSpatialTransformerDescriptor(&samplerDesc)) ;
    samplerDescInitialized = true ;
    CHECK(cudnnSetSpatialTransformerNdDescriptor( samplerDesc,
                                                  CUDNN_SAMPLER_BILINEAR,
                                                  cudnnDataType,
                                                  4,
                                                  dimOut));
    /* do the work */
    {
      type alpha = 1.0f;
      type beta = 0.0f;
      cudnnSpatialTfSamplerForward( handle,
                                    samplerDesc,
                                    &alpha,
                                    dataDesc, data.getMemory(),
                                    grid.getMemory(),
                                    &beta,
                                    outputDesc, output.getMemory());
    }

    /* cleanup */
  done:
    if (samplerDescInitialized) { cudnnDestroySpatialTransformerDescriptor(samplerDesc) ; }
    if (dataDescInitialized) { cudnnDestroyTensorDescriptor(dataDesc) ; }
    if (outputDescInitialized) { cudnnDestroyTensorDescriptor(outputDesc) ; }
    return context.passError(error, __func__) ;
  }

  /* ---------------------------------------------------------------- */
  /*                                   bilinearsampler_backward_cudnn */
  /* ---------------------------------------------------------------- */
  template<vl::Type dataType>
  vl::Error
  vl::impl::nnbilinearsampler_cudnn<dataType>::backward( Context& context,
                                                         Tensor derData,
                                                         Tensor derGrid,
                                                         Tensor data,
                                                         Tensor grid,
                                                         Tensor derOutput)
  {
    
    typedef typename DataTypeTraits<dataType>::type type ;

    /* no derDataDesc needed as same as dataDesc <-- nice! */
    cudnnTensorDescriptor_t dataDesc, derOutputDesc ;
    cudnnSpatialTransformerDescriptor_t samplerDesc ;
    bool dataDescInitialized = false ;
    bool derOutputDescInitialized = false ;
    bool samplerDescInitialized = false ;

    // get the sizes:
    int inCardinality = data.getSize();
    int inDepth = data.getDepth();
    int inHeight = data.getHeight();
    int inWidth = data.getWidth();

    int outCardinality = derOutput.getSize();
    int outDepth = derOutput.getDepth();
    int outWidth = derOutput.getWidth();
    int outHeight = derOutput.getHeight();

    int dimOut[4] = { outCardinality, outDepth, outWidth, outHeight };

    cudnnDataType_t cudnnDataType = DataTypeToCudnn<dataType>::id ;
    vl::Type dynDataType = derOutput.getDataType() ;
    assert(dynDataType == dataType) ;

    cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
    vl::Error error = vl::vlSuccess ;
    cudnnHandle_t handle ;

    // Get CuDNN
    CHECK(context.getCudaHelper().getCudnnHandle(&handle)) ;

    // Get tensor descriptors:
    CHECK(cudnnCreateTensorDescriptor(&derOutputDesc)) ;
    derOutputDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptorEx(derOutputDesc,
                                       cudnnDataType,
                                       outCardinality, outDepth, outWidth, outHeight, // sizes: n,c,w,h
                                       outHeight * outWidth * outDepth, //strides
                                       outHeight * outWidth,
                                       outHeight,
                                       1)) ;

    CHECK(cudnnCreateTensorDescriptor(&dataDesc)) ;
    dataDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptorEx(dataDesc,
                                       cudnnDataType,
                                       inCardinality, inDepth, inWidth, inHeight, // sizes: n,c,w,h
                                       inHeight * inWidth * inDepth, //strides
                                       inHeight * inWidth,
                                       inHeight,
                                       1)) ;

    // Get bilinear-sampler descriptor:
    CHECK(cudnnCreateSpatialTransformerDescriptor(&samplerDesc)) ;
    samplerDescInitialized = true ;
    CHECK(cudnnSetSpatialTransformerNdDescriptor( samplerDesc,
                                                  CUDNN_SAMPLER_BILINEAR,
                                                  cudnnDataType,
                                                  4,
                                                  dimOut));
    /* do the work */
    {
      type alpha = 1.0f;
      type beta = 0.0f;
      cudnnSpatialTfSamplerBackward(  handle,
                                      samplerDesc,
                                      &alpha,
                                      dataDesc, data.getMemory(),
                                      &beta,
                                      dataDesc, derData.getMemory(),
                                      &alpha,
                                      derOutputDesc, derOutput.getMemory(),
                                      grid.getMemory(),
                                      &beta,
                                      derGrid.getMemory() );
    }

  /* cleanup */
  done:
    if (samplerDescInitialized) { cudnnDestroySpatialTransformerDescriptor(samplerDesc) ; }
    if (dataDescInitialized) { cudnnDestroyTensorDescriptor(dataDesc) ; }
    if (derOutputDescInitialized) { cudnnDestroyTensorDescriptor(derOutputDesc) ; }
    return context.passError(error, __func__) ;
  }
}}

// Instantiations
template struct vl::impl::nnbilinearsampler_cudnn<vl::vlTypeFloat> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::nnbilinearsampler_cudnn<vl::vlTypeDouble> ;
#endif
