// @file nnbias_cudnn.cu
// @brief biasolution block CuDNN-based implementation.
// @author Andrea Vedaldi

/*
Copyright (C) 2015-17 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnbias.hpp"
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
struct BiasForwardCudnn
{
  vl::ErrorCode operator()(Bias & op,
                           Tensor &output, double outputMult,
                           Tensor const &input, double inputMult,
                           Tensor const &bias, double biasMult)
  {
    typedef typename DataTypeTraits<dataType>::type type ;

    cudnnTensorDescriptor_t outputDesc, biasDesc, dataDesc ;
    bool outputDescInitialized = false ;
    bool biasDescInitialized = false ;
    bool dataDescInitialized = false ;

    cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
    vl::ErrorCode error = vl::VLE_Success ;
    cudnnHandle_t handle ;

    // Get CuDNN
    CHECK(op.context.getCudaHelper().getCudnnHandle(&handle)) ;

    // Get output tensor descripotr
    assert(output) ;
    CHECK(cudnnCreateTensorDescriptor(&outputDesc)) ;
    outputDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptor(outputDesc,
                                     CUDNN_TENSOR_NCHW,
                                     DataTypeToCudnn<dataType>::dataType,
                                     output.getSize(), // sizes
                                     output.getDepth(),
                                     output.getWidth(),
                                     output.getHeight())) ;

    if (bias) {
      CHECK(cudnnCreateTensorDescriptor(&biasDesc)) ;
      biasDescInitialized = true ;
      CHECK(cudnnSetTensor4dDescriptor(biasDesc,
                                       CUDNN_TENSOR_NCHW,
                                       DataTypeToCudnn<dataType>::dataType,
                                       1,
                                       bias.getNumElements(),
                                       1,
                                       1)) ;

      type alpha = biasMult ;
      type beta = outputMult ;
#if (CUDNN_VERSION < 4000)
      CHECK(cudnnAddTensor(handle,
                           CUDNN_ADD_SAME_C,
                           &alpha,
                           biasDesc, bias.getMemory(),
                           &beta,
                           outputDesc, output.getMemory())) ;
#else
      CHECK(cudnnAddTensor(handle,
                           &alpha,
                           biasDesc, bias.getMemory(),
                           &beta,
                           outputDesc, output.getMemory())) ;
#endif
      outputMult = 1 ;
    }

    if (input) {
      CHECK(cudnnCreateTensorDescriptor(&dataDesc)) ;
      dataDescInitialized = true ;
      CHECK(cudnnSetTensor4dDescriptor(dataDesc,
                                       CUDNN_TENSOR_NCHW,
                                       DataTypeToCudnn<dataType>::dataType,
                                       input.getSize(),
                                       input.getDepth(),
                                       input.getWidth(),
                                       input.getHeight())) ;

      type alpha = inputMult ;
      type beta = outputMult ;
#if (CUDNN_VERSION < 4000)
      CHECK(cudnnAddTensor(handle,
                           CUDNN_ADD_FULL_TENSOR,
                           &alpha,
                           dataDesc, input.getMemory(),
                           &beta,
                           outputDesc, output.getMemory()));
#else
      CHECK(cudnnAddTensor(handle,
                           &alpha,
                           dataDesc, input.getMemory(),
                           &beta,
                           outputDesc, output.getMemory()));
#endif
    }

    /* cleanup */
  done:
    if (dataDescInitialized) { cudnnDestroyTensorDescriptor(dataDesc) ; }
    if (biasDescInitialized) { cudnnDestroyTensorDescriptor(biasDesc) ; }
    if (outputDescInitialized) { cudnnDestroyTensorDescriptor(outputDesc) ; }
    return op.context.passError(error, __func__) ;
  }
} ;


// -------------------------------------------------------------------
//                                                            Backward
// -------------------------------------------------------------------

template<DataType dataType>
struct BiasBackwardCudnn
{
  vl::ErrorCode operator()(Bias &op,
                           Tensor &derInput, double derInputMult,
                           Tensor &derBias, double derBiasMult,
                           double inputMult, double biasMult,
                           Tensor const &derOutput)
  {
    typedef typename DataTypeTraits<dataType>::type type ;

    /* no derInputDesc needed as same as dataDesc */
    cudnnTensorDescriptor_t derInputDesc, derBiasDesc, derOutputDesc ;
    bool derInputDescInitialized = false ;
    bool derBiasDescInitialized = false ;
    bool derOutputDescInitialized = false ;

    cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
    vl::ErrorCode error = vl::VLE_Success ;
    cudnnHandle_t handle ;

    // Get CuDNN
    CHECK(op.context.getCudaHelper().getCudnnHandle(&handle)) ;

    // Must have derOutput for all derivatives
    assert(derOutput) ;
    CHECK(cudnnCreateTensorDescriptor(&derOutputDesc)) ;
    derOutputDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptor(derOutputDesc,
                                     CUDNN_TENSOR_NCHW,
                                     DataTypeToCudnn<dataType>::dataType,
                                     derOutput.getSize(), // sizes
                                     derOutput.getDepth(),
                                     derOutput.getWidth(),
                                     derOutput.getHeight())) ;

    // for derivatives w.r.t. bias
    if (derBias) {
      CHECK(cudnnCreateTensorDescriptor(&derBiasDesc)) ;
      derBiasDescInitialized = true ;
      CHECK(cudnnSetTensor4dDescriptor(derBiasDesc,
                                       CUDNN_TENSOR_NCHW,
                                       DataTypeToCudnn<dataType>::dataType,
                                       1,
                                       derBias.getNumElements(),
                                       1,
                                       1)) ;

      type alpha = biasMult ;
      type beta = derBiasMult ;
      CHECK(cudnnConvolutionBackwardBias
            (handle,
             &alpha,
             derOutputDesc, (type const*)derOutput.getMemory(),
             &beta,
             derBiasDesc, (type*)derBias.getMemory())) ;
    }

    if (derInput) {
      CHECK(cudnnCreateTensorDescriptor(&derInputDesc)) ;
      derInputDescInitialized = true ;
      CHECK(cudnnSetTensor4dDescriptor(derInputDesc,
                                       CUDNN_TENSOR_NCHW,
                                       DataTypeToCudnn<dataType>::dataType,
                                       derInput.getSize(),
                                       derInput.getDepth(),
                                       derInput.getWidth(),
                                       derInput.getHeight())) ;
      type alpha = inputMult ;
      type beta = derInputMult ;
#if (CUDNN_VERSION < 4000)
      CHECK(cudnnAddTensor(handle,
                           CUDNN_ADD_SAME_C,
                           &alpha,
                           biasDesc, bias.getMemory(),
                           &beta,
                           derInputDesc, derInput.getMemory())) ;
#else
      CHECK(cudnnAddTensor(handle,
                           &alpha,
                           derOutputDesc, derOutput.getMemory(),
                           &beta,
                           derInputDesc, derInput.getMemory())) ;
#endif
    }

  done:
    if (derOutputDescInitialized) { cudnnDestroyTensorDescriptor(derOutputDesc) ; }
    if (derBiasDescInitialized) { cudnnDestroyTensorDescriptor(derBiasDesc) ; }
    if (derInputDescInitialized) { cudnnDestroyTensorDescriptor(derInputDesc) ; }
    return op.context.passError(error, __func__) ;
  }
} ;
