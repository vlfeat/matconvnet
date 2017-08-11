// @file nnpooling_cudnn.cu
// @brief Pooling layer CuDNN.
// @author Andrea Vedaldi

/*
Copyright (C) 2015-17 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnpooling.hpp"
#include "datacu.hpp"
#include "impl/cudnnhelper.hpp"
#include <cassert>

using namespace std ;
using namespace vl ;
using namespace vl::nn ;
using namespace vl::impl ;

#define CHECK(x) \
{ \
cudnnError = x ; \
if (cudnnError != CUDNN_STATUS_SUCCESS) { \
error = op.context.setError(op.context.getCudaHelper().catchCudnnError(cudnnError, \
STRINGIZE(__LINE__) ":" STRINGIZE(__FILE__))) ; \
goto done ; \
} }

// -------------------------------------------------------------------
//                                                             Forward
// -------------------------------------------------------------------

template<DataType dataType>
struct PoolingForwardCudnn
{
  vl::ErrorCode operator()(Pooling &op,
                           Tensor &output,
                           Tensor const &input)
  {
    assert(output) ;
    assert(input) ;

    typedef typename DataTypeTraits<dataType>::type type ;

    cudnnTensorDescriptor_t outputDesc, inputDesc ;
    cudnnPoolingDescriptor_t poolingDesc ;
    bool outputDescInitialized = false ;
    bool inputDescInitialized = false ;
    bool poolingDescInitialized = false ;

    if (op.padLeft != op.padRight) return vl::VLE_Unsupported ;
    if (op.padTop != op.padBottom) return vl::VLE_Unsupported ;

    if (op.method == Pooling::Average && (op.padLeft > 0 | op.padRight > 0)) {
      // CuDNN bug? Skip.
      return vl::VLE_Unsupported ;
    }

    cudnnDataType_t cudnnDataType = DataTypeToCudnn<dataType>::dataType ;
    vl::DataType dynDataType = output.getDataType() ;
    assert(dynDataType == dataType) ;

    cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
    vl::ErrorCode error = vl::VLE_Success ;
    cudnnHandle_t handle ;

    // Get CuDNN.
    CHECK(op.context.getCudaHelper().getCudnnHandle(&handle)) ;

    // Get tensor descriptors.
    CHECK(cudnnCreateTensorDescriptor(&outputDesc)) ;
    outputDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptor(outputDesc,
                                     CUDNN_TENSOR_NCHW,
                                     cudnnDataType,
                                     output.getSize(),
                                     output.getDepth(),
                                     output.getWidth(),
                                     output.getHeight())) ;

    CHECK(cudnnCreateTensorDescriptor(&inputDesc)) ;
    inputDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptor(inputDesc,
                                     CUDNN_TENSOR_NCHW,
                                     cudnnDataType,
                                     input.getSize(),
                                     input.getDepth(),
                                     input.getWidth(),
                                     input.getHeight())) ;

    CHECK(cudnnCreatePoolingDescriptor(&poolingDesc)) ;
    poolingDescInitialized = true ;
    CHECK(cudnnSetPooling2dDescriptor(poolingDesc,
                                      (op.method == Pooling::Average) ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING : CUDNN_POOLING_MAX,
                                      IF_CUDNN_GE5(CUDNN_NOT_PROPAGATE_NAN COMMA)
                                      op.poolWidth, op.poolHeight,
                                      op.padLeft, op.padTop,
                                      op.strideX, op.strideY)) ;

    // Apply operator.
    {
      type alpha = 1.0f ;
      type beta = 0.0f ;
      CHECK(cudnnPoolingForward(handle,
                                poolingDesc,
                                &alpha,
                                inputDesc, input.getMemory(),
                                &beta,
                                outputDesc, output.getMemory())) ;
    }

    // Finish.
  done:
    if (poolingDescInitialized) { cudnnDestroyPoolingDescriptor(poolingDesc) ; }
    if (inputDescInitialized) { cudnnDestroyTensorDescriptor(inputDesc) ; }
    if (outputDescInitialized) { cudnnDestroyTensorDescriptor(outputDesc) ; }
    return op.context.passError(error, "nnpooling_cudnn::forward") ;
  }
} ;

// -------------------------------------------------------------------
//                                                            Backward
// -------------------------------------------------------------------

template<DataType dataType>
struct PoolingBackwardCudnn
{
  vl::ErrorCode operator()(Pooling &op,
                           Tensor &derInput,
                           Tensor const &input,
                           Tensor const &derOutput)
  {
    assert(derInput) ;
    assert(input) ;
    assert(derOutput) ;

    vl::ErrorCode error = VLE_Success ;
    typedef typename DataTypeTraits<dataType>::type type ;

    cudnnTensorDescriptor_t derOutputDesc, inputDesc ;
    cudnnPoolingDescriptor_t poolingDesc ;
    bool derOutputDescInitialized = false ;
    bool inputDescInitialized = false ;
    bool poolingDescInitialized = false ;

    if (op.padLeft != op.padRight) return vl::VLE_Unsupported ;
    if (op.padTop != op.padBottom) return vl::VLE_Unsupported ;

    if (op.method == Pooling::Average && (op.padLeft > 0 | op.padRight > 0)) {
      // CuDNN bug? Skip.
      return vl::VLE_Unsupported ;
    }

    cudnnDataType_t cudnnDataType = DataTypeToCudnn<dataType>::dataType ;
    vl::DataType dynDataType = derInput.getDataType() ;
    assert(dynDataType == dataType) ;

    cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
    cudnnHandle_t handle ;
    Tensor output ;

    // CuDNN requires the output of the layer, so we recompute it here.
    size_t outputDataSize = derOutput.getNumElements() * sizeof(type) ;
    type * outputData = (type*)op.context.getWorkspace
    (vl::VLDT_GPU, outputDataSize) ;
    if (outputData == NULL) {
      error = VLE_OutOfMemory ;
      goto done ;
    }
    output = Tensor(derOutput, dataType, VLDT_GPU, outputData, outputDataSize) ;
    error = PoolingForwardCudnn<dataType>()(op,output,input) ;
    if (error != VLE_Success) {
      goto done ;
    }

    // Get CuDNN.
    CHECK(op.context.getCudaHelper().getCudnnHandle(&handle)) ;

    // Get tensor descripotrs.
    CHECK(cudnnCreateTensorDescriptor(&derOutputDesc)) ;
    derOutputDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptor(derOutputDesc,
                                     CUDNN_TENSOR_NCHW,
                                     cudnnDataType,
                                     derOutput.getSize(),
                                     derOutput.getDepth(),
                                     derOutput.getWidth(),
                                     derOutput.getHeight())) ;

    CHECK(cudnnCreateTensorDescriptor(&inputDesc)) ;
    inputDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptor(inputDesc,
                                     CUDNN_TENSOR_NCHW,
                                     cudnnDataType,
                                     input.getSize(),
                                     input.getDepth(),
                                     input.getWidth(),
                                     input.getHeight())) ;

    CHECK(cudnnCreatePoolingDescriptor(&poolingDesc)) ;
    poolingDescInitialized = true ;
    CHECK(cudnnSetPooling2dDescriptor(poolingDesc,
                                      (op.method == Pooling::Average) ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING : CUDNN_POOLING_MAX,
                                      IF_CUDNN_GE5(CUDNN_NOT_PROPAGATE_NAN COMMA)
                                      op.poolWidth, op.poolHeight,
                                      op.padLeft, op.padTop,
                                      op.strideX, op.strideY)) ;

    // Apply operator.
    {
      type alpha = 1.0f ;
      type beta = 0.0f ;
      CHECK(cudnnPoolingBackward(handle,
                                 poolingDesc,
                                 &alpha,
                                 derOutputDesc, outputData,
                                 derOutputDesc, (type const*)derOutput.getMemory(),
                                 inputDesc, (type const*)input.getMemory(),
                                 &beta,
                                 inputDesc, (type*)derInput.getMemory())) ;
    }

    // Finish.
  done:
    if (poolingDescInitialized) { cudnnDestroyPoolingDescriptor(poolingDesc) ; }
    if (inputDescInitialized) { cudnnDestroyTensorDescriptor(inputDesc) ; }
    if (derOutputDescInitialized) { cudnnDestroyTensorDescriptor(derOutputDesc) ; }
    return op.context.passError(error, __func__) ;
  }
} ;



