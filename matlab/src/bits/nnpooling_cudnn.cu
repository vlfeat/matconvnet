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

template <typename F>
class deferred {
public:
  explicit deferred(F f) : f(std::move(f)) { }
  deferred(const deferred&) = delete ;
  deferred(deferred&& d) : f(std::move(d.f)) { d.armed = false ; }
  deferred& operator= (deferred const &) = delete ;
  deferred& operator= (deferred&&) = delete ;
  ~deferred() { f() ; }

private:
  F f ;
  bool armed = true ;
} ;

template <typename F>
deferred<F> defer(F f) {
  return deferred<F>(std::move(f)) ;
}

#define CHECK(x) \
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
struct PoolingForwardCudnn
{
  vl::ErrorCode operator()(Pooling const &op,
                           Tensor &output,
                           Tensor const &input)
  {
    typedef typename DataTypeTraits<dataType>::type type ;
    if (op.getPadding(2) != op.getPadding(3)) return vl::VLE_Unsupported ;
    if (op.getPadding(0) != op.getPadding(1)) return vl::VLE_Unsupported ;
    if (op.getMethod() == Pooling::Average && (op.getPadding(2) > 0 | op.getPadding(3) > 0)) {
      // CuDNN bug? Skip.
      return vl::VLE_Unsupported ;
    }

    static const std::string signature = std::string("PoolingForward[CuDNN")
    + DeviceTypeTraits<VLDT_GPU>::name + "," + DataTypeTraits<dataType>::name + "]" ;
    VLLOG(op,1) << signature ;

    assert(output) ;
    assert(input) ;

    // Get CuDNN.
    cudnnHandle_t handle ;
    CHECK(op.getContext().getCudaHelper().getCudnnHandle(&handle)) ;

    // Get tensor descriptors.
    CudnnTensorDescriptor outputDesc, inputDesc ;
    CHECK(outputDesc.init(dataType,output.getShape())) ;
    CHECK(inputDesc.init(dataType,input.getShape())) ;

    // Get pooling descriptor.
    cudnnPoolingDescriptor_t poolingDesc ;
    CHECK(cudnnCreatePoolingDescriptor(&poolingDesc)) ;
    auto poolingDescDeleter = defer([&]{cudnnDestroyPoolingDescriptor(poolingDesc);}) ;
    CHECK(cudnnSetPooling2dDescriptor(poolingDesc,
                                      (op.getMethod() == Pooling::Average) ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING : CUDNN_POOLING_MAX,
                                      IF_CUDNN_GE5(CUDNN_NOT_PROPAGATE_NAN COMMA)
                                      (int)op.getShape(1), (int)op.getShape(0),
                                      (int)op.getPadding(2), (int)op.getPadding(0),
                                      (int)op.getStride(1), (int)op.getStride(0))) ;

    // Perform calculation.
    type alpha = 1.0f ;
    type beta = 0.0f ;
    CHECK(cudnnPoolingForward(handle,
                              poolingDesc,
                              &alpha,
                              inputDesc.get(), input.getMemory(),
                              &beta,
                              outputDesc.get(), output.getMemory())) ;

    return VLE_Success ;
  }
} ;

// -------------------------------------------------------------------
//                                                            Backward
// -------------------------------------------------------------------

template<DataType dataType>
struct PoolingBackwardCudnn
{
  vl::ErrorCode operator()(Pooling const &op,
                           Tensor &derInput,
                           Tensor const &input,
                           Tensor const &derOutput)
  {
    typedef typename DataTypeTraits<dataType>::type type ;

    if (op.getPadding(2) != op.getPadding(3)) return vl::VLE_Unsupported ;
    if (op.getPadding(0) != op.getPadding(1)) return vl::VLE_Unsupported ;

    if (op.getMethod() == Pooling::Average && (op.getPadding(2) > 0 | op.getPadding(3) > 0)) {
      // CuDNN bug? Skip.
      return vl::VLE_Unsupported ;
    }

    static const std::string signature = std::string("PoolingBackward[CuDNN")
    + DeviceTypeTraits<VLDT_GPU>::name + "," + DataTypeTraits<dataType>::name + "]" ;
    VLLOG(op,1) << signature ;

    assert(derInput) ;
    assert(input) ;
    assert(derOutput) ;

    // CuDNN requires the output of the layer, so we recompute it here.
    size_t outputDataSize = (size_t)derOutput.getNumElements() * sizeof(type) ;
    type * outputData = (type*)op.getContext().getWorkspace
    (vl::VLDT_GPU, outputDataSize) ;
    if (!outputData) {
      return op.getContext().setError
      (VLE_OutOfMemory, "PoolingBackward: out of memory.") ;
    }

    auto output = Tensor(derOutput, dataType, VLDT_GPU, outputData, outputDataSize) ;
    auto error = PoolingForwardCudnn<dataType>()(op,output,input) ;
    if (error != VLE_Success) {
      return op.getContext().passError(error,signature.c_str()) ;
    }

    // Get CuDNN.
    cudnnHandle_t handle ;
    CHECK(op.getContext().getCudaHelper().getCudnnHandle(&handle)) ;

    // Get tensor descripotrs.
    CudnnTensorDescriptor derOutputDesc, inputDesc ;
    CHECK(derOutputDesc.init(dataType,derOutput.getShape())) ;
    CHECK(inputDesc.init(dataType,input.getShape())) ;

    // Get pooling descriptor.
    cudnnPoolingDescriptor_t poolingDesc ;
    CHECK(cudnnCreatePoolingDescriptor(&poolingDesc)) ;
    auto poolingDescDeleter = defer([&]{cudnnDestroyPoolingDescriptor(poolingDesc);}) ;
    CHECK(cudnnSetPooling2dDescriptor(poolingDesc,
                                      (op.getMethod() == Pooling::Average) ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING : CUDNN_POOLING_MAX,
                                      IF_CUDNN_GE5(CUDNN_NOT_PROPAGATE_NAN COMMA)
                                      (int)op.getShape(1), (int)op.getShape(0),
                                      (int)op.getPadding(2), (int)op.getPadding(0),
                                      (int)op.getStride(1), (int)op.getStride(0))) ;

    // Perform calculation.
    type alpha = 1.0f ;
    type beta = 0.0f ;
    CHECK(cudnnPoolingBackward(handle,
                               poolingDesc,
                               &alpha,
                               derOutputDesc.get(), outputData,
                               derOutputDesc.get(), (type const*)derOutput.getMemory(),
                               inputDesc.get(), (type const*)input.getMemory(),
                               &beta,
                               inputDesc.get(), (type*)derInput.getMemory())) ;

    return VLE_Success ;
  }
} ;



