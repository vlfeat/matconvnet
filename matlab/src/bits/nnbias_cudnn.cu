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
error = op.getContext().setError(op.getContext().getCudaHelper().catchCudnnError(cudnnError, \
STRINGIZE(__LINE__) ":" STRINGIZE(__FILE__))) ; \
goto done ; \
} }

// -------------------------------------------------------------------
//                                                             Forward
// -------------------------------------------------------------------

template<DataType dataType>
struct BiasForwardCudnn
{
  vl::ErrorCode operator()(Bias const& op,
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
    CHECK(op.getContext().getCudaHelper().getCudnnHandle(&handle)) ;

    // Get output tensor descripotr
    assert(output) ;
    CHECK(cudnnCreateTensorDescriptor(&outputDesc)) ;
    outputDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptor(outputDesc,
                                     CUDNN_TENSOR_NCHW,
                                     DataTypeToCudnn<dataType>::dataType,
                                     (int)output.getSize(), // sizes
                                     (int)output.getDepth(),
                                     (int)output.getWidth(),
                                     (int)output.getHeight())) ;

    if (bias) {
      CHECK(cudnnCreateTensorDescriptor(&biasDesc)) ;
      biasDescInitialized = true ;
      CHECK(cudnnSetTensor4dDescriptor(biasDesc,
                                       CUDNN_TENSOR_NCHW,
                                       DataTypeToCudnn<dataType>::dataType,
                                       1,
                                       (int)bias.getNumElements(),
                                       1,
                                       1)) ;

      auto alpha = static_cast<type>(biasMult) ;
      auto beta = static_cast<type>(outputMult) ;
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
                                       (int)input.getSize(),
                                       (int)input.getDepth(),
                                       (int)input.getWidth(),
                                       (int)input.getHeight())) ;

      auto alpha = static_cast<type>(biasMult) ;
      auto beta = static_cast<type>(outputMult) ;
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
    return op.getContext().passError(error, __func__) ;
  }
} ;


// -------------------------------------------------------------------
//                                                            Backward
// -------------------------------------------------------------------

template<DataType dataType>
struct BiasBackwardCudnn
{
  vl::ErrorCode operator()(Bias const &op,
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
    CHECK(op.getContext().getCudaHelper().getCudnnHandle(&handle)) ;

    // Must have derOutput for all derivatives
    assert(derOutput) ;
    CHECK(cudnnCreateTensorDescriptor(&derOutputDesc)) ;
    derOutputDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptor(derOutputDesc,
                                     CUDNN_TENSOR_NCHW,
                                     DataTypeToCudnn<dataType>::dataType,
                                     (int)derOutput.getSize(), // sizes
                                     (int)derOutput.getDepth(),
                                     (int)derOutput.getWidth(),
                                     (int)derOutput.getHeight())) ;

    // for derivatives w.r.t. bias
    if (derBias) {
      CHECK(cudnnCreateTensorDescriptor(&derBiasDesc)) ;
      derBiasDescInitialized = true ;
      CHECK(cudnnSetTensor4dDescriptor(derBiasDesc,
                                       CUDNN_TENSOR_NCHW,
                                       DataTypeToCudnn<dataType>::dataType,
                                       1,
                                       (int)derBias.getNumElements(),
                                       1,
                                       1)) ;

      auto alpha = static_cast<type>(biasMult) ;
      auto beta = static_cast<type>(derBiasMult) ;
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
                                       (int)derInput.getSize(),
                                       (int)derInput.getDepth(),
                                       (int)derInput.getWidth(),
                                       (int)derInput.getHeight())) ;
      auto alpha = static_cast<type>(biasMult) ;
      auto beta = static_cast<type>(derBiasMult) ;
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
    return op.getContext().passError(error, __func__) ;
  }
} ;
