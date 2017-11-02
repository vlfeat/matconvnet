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
    static const std::string signature = std::string("BiasForward[CuDNN,")
    + DeviceTypeTraits<VLDT_GPU>::name + "," + DataTypeTraits<dataType>::name + "]" ;

    typedef typename DataTypeTraits<dataType>::type type ;

    CudnnTensorDescriptor outputDesc, biasDesc, dataDesc ;
    cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
    vl::ErrorCode error = vl::VLE_Success ;
    cudnnHandle_t handle ;

    // Get CuDNN.
    CHECK(op.getContext().getCudaHelper().getCudnnHandle(&handle)) ;

    // Get output tensor descripotr.
    assert(output) ;
    CHECK(outputDesc.init(dataType, output.getShape())) ;

    if (bias) {
      CHECK(biasDesc.init(dataType,{1,1,bias.getNumElements(),1})) ;
      auto alpha = static_cast<type>(biasMult) ;
      auto beta = static_cast<type>(outputMult) ;
#if (CUDNN_VERSION < 4000)
      CHECK(cudnnAddTensor(handle,
                           CUDNN_ADD_SAME_C,
                           &alpha,
                           biasDesc.get(), bias.getMemory(),
                           &beta,
                           outputDesc.get(), output.getMemory())) ;
#else
      CHECK(cudnnAddTensor(handle,
                           &alpha,
                           biasDesc.get(), bias.getMemory(),
                           &beta,
                           outputDesc.get(), output.getMemory())) ;
#endif
      outputMult = 1 ;
    }

    if (input) {
      CHECK(dataDesc.init(dataType,input.getShape())) ;
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
                           dataDesc.get(), input.getMemory(),
                           &beta,
                           outputDesc.get(), output.getMemory())) ;
#endif
    }

    /* cleanup */
  done:
    return op.getContext().passError(error,signature.c_str()) ;
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
    static const std::string signature = std::string("BiasBackward[CuDNN,")
    + DeviceTypeTraits<VLDT_GPU>::name + "," + DataTypeTraits<dataType>::name + "]" ;

    typedef typename DataTypeTraits<dataType>::type type ;

    // no derInputDesc needed as same as dataDesc.
    CudnnTensorDescriptor derInputDesc, derBiasDesc, derOutputDesc ;
    cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
    vl::ErrorCode error = vl::VLE_Success ;
    cudnnHandle_t handle ;

    // Get CuDNN.
    CHECK(op.getContext().getCudaHelper().getCudnnHandle(&handle)) ;

    // Must have derOutput for all derivatives
    assert(derOutput) ;
    CHECK(derOutputDesc.init(dataType,derOutput.getShape())) ;

    if (derBias) {
      CHECK(derBiasDesc.init(dataType,{1,1,derBias.getNumElements(),1})) ;
      auto alpha = static_cast<type>(biasMult) ;
      auto beta = static_cast<type>(derBiasMult) ;
      CHECK(cudnnConvolutionBackwardBias
            (handle,
             &alpha,
             derOutputDesc.get(), (type const*)derOutput.getMemory(),
             &beta,
             derBiasDesc.get(), (type*)derBias.getMemory())) ;
    }

    if (derInput) {
      CHECK(derInputDesc.init(dataType,derInput.getShape())) ;
      auto alpha = static_cast<type>(biasMult) ;
      auto beta = static_cast<type>(derBiasMult) ;
#if (CUDNN_VERSION < 4000)
      CHECK(cudnnAddTensor(handle,
                           CUDNN_ADD_SAME_C,
                           &alpha,
                           biasDesc.get(), bias.getMemory(),
                           &beta,
                           derInputDesc.get(), derInput.getMemory())) ;
#else
      CHECK(cudnnAddTensor(handle,
                           &alpha,
                           derOutputDesc.get(), derOutput.getMemory(),
                           &beta,
                           derInputDesc.get(), derInput.getMemory())) ;
#endif
    }

  done:
    return op.getContext().passError(error,signature.c_str()) ;
  }
} ;
