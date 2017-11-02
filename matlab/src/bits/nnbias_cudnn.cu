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

#define CKCUDNN(x) \
{ \
cudnnStatus_t cudnnError = (x) ; \
if (cudnnError != CUDNN_STATUS_SUCCESS) { \
return op.getContext().setError(op.getContext().getCudaHelper().catchCudnnError(cudnnError, \
STRINGIZE(__LINE__) ":" STRINGIZE(__FILE__))) ; \
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

    static const std::string signature = std::string("BiasForward[CuDNN,")
    + DeviceTypeTraits<VLDT_GPU>::name + "," + DataTypeTraits<dataType>::name + "]" ;
    VLLOG(op,1) << signature.c_str() ;

    // Get CuDNN.
    cudnnHandle_t handle ;
    CKCUDNN(op.getContext().getCudaHelper().getCudnnHandle(&handle)) ;

    // Get output tensor descripotr.
    assert(output) ;
    CudnnTensorDescriptor outputDesc ;
    CKCUDNN(outputDesc.init(dataType, output.getShape())) ;

    if (bias) {
      CudnnTensorDescriptor biasDesc ;
      CKCUDNN(biasDesc.init(dataType,{1,1,bias.getNumElements(),1})) ;
      auto alpha = static_cast<type>(biasMult) ;
      auto beta = static_cast<type>(outputMult) ;
#if (CUDNN_VERSION < 4000)
      CKCUDNN(cudnnAddTensor(handle,
                             CUDNN_ADD_SAME_C,
                             &alpha,
                             biasDesc.get(), bias.getMemory(),
                             &beta,
                             outputDesc.get(), output.getMemory())) ;
#else
      CKCUDNN(cudnnAddTensor(handle,
                             &alpha,
                             biasDesc.get(), bias.getMemory(),
                             &beta,
                             outputDesc.get(), output.getMemory())) ;
#endif
      outputMult = 1 ;
    }

    if (input) {
      CudnnTensorDescriptor inputDesc ;
      CKCUDNN(inputDesc.init(dataType,input.getShape())) ;
      auto alpha = static_cast<type>(biasMult) ;
      auto beta = static_cast<type>(outputMult) ;
#if (CUDNN_VERSION < 4000)
      CKCUDNN(cudnnAddTensor(handle,
                             CUDNN_ADD_FULL_TENSOR,
                             &alpha,
                             inputDesc, input.getMemory(),
                             &beta,
                             outputDesc, output.getMemory()));
#else
      CKCUDNN(cudnnAddTensor(handle,
                             &alpha,
                             inputDesc.get(), input.getMemory(),
                             &beta,
                             outputDesc.get(), output.getMemory())) ;
#endif
    }
    return VLE_Success ;
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
    static const std::string signature = std::string("BiasBackward[CuDNN,")
    + DeviceTypeTraits<VLDT_GPU>::name + "," + DataTypeTraits<dataType>::name + "]" ;
    VLLOG(op,1) << signature.c_str() ;

    // Get CuDNN.
    cudnnHandle_t handle ;
    CKCUDNN(op.getContext().getCudaHelper().getCudnnHandle(&handle)) ;

    // Must have derOutput for all derivatives
    assert(derOutput) ;
    CudnnTensorDescriptor derOutputDesc ;
    CKCUDNN(derOutputDesc.init(dataType,derOutput.getShape())) ;

    if (derBias) {
      CudnnTensorDescriptor derBiasDesc ;
      CKCUDNN(derBiasDesc.init(dataType,{1,1,derBias.getNumElements(),1})) ;
      auto alpha = static_cast<type>(biasMult) ;
      auto beta = static_cast<type>(derBiasMult) ;
      CKCUDNN(cudnnConvolutionBackwardBias
              (handle,
               &alpha,
               derOutputDesc.get(), (type const*)derOutput.getMemory(),
               &beta,
               derBiasDesc.get(), (type*)derBias.getMemory())) ;
    }

    if (derInput) {
      CudnnTensorDescriptor derInputDesc ;
      CKCUDNN(derInputDesc.init(dataType,derInput.getShape())) ;
      auto alpha = static_cast<type>(biasMult) ;
      auto beta = static_cast<type>(derBiasMult) ;
#if (CUDNN_VERSION < 4000)
      CKCUDNN(cudnnAddTensor(handle,
                             CUDNN_ADD_SAME_C,
                             &alpha,
                             biasDesc.get(), bias.getMemory(),
                             &beta,
                             derInputDesc.get(), derInput.getMemory())) ;
#else
      CKCUDNN(cudnnAddTensor(handle,
                             &alpha,
                             derOutputDesc.get(), derOutput.getMemory(),
                             &beta,
                             derInputDesc.get(), derInput.getMemory())) ;
#endif
    }
    return VLE_Success ;
  }
} ;
