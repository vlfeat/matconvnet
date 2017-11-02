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
#include <cassert>

#define COMMA ,

#if (CUDNN_VERSION >= 6000)
#define IF_CUDNN_GE6(x) x
#else
#define IF_CUDNN_GE6(x)
#endif

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

#define CKCUDNN(x) \
{ \
cudnnStatus_t cudnnError = (x) ; \
if (cudnnError != CUDNN_STATUS_SUCCESS) { \
return op.getContext().setError(op.getContext().getCudaHelper().catchCudnnError(cudnnError, \
STRINGIZE(__LINE__) ":" STRINGIZE(__FILE__))) ; \
} }

namespace vl { namespace impl {

  template <vl::DataType dataType> struct DataTypeToCudnn { } ;
  template <> struct DataTypeToCudnn<vl::VLDT_Float> { static cudnnDataType_t const dataType = CUDNN_DATA_FLOAT ; } ;
  template <> struct DataTypeToCudnn<vl::VLDT_Double> { static cudnnDataType_t const dataType = CUDNN_DATA_DOUBLE ; } ;

  inline cudnnDataType_t dataTypeToCudnn(vl::DataType dataType)
  {
    switch (dataType) {
      case VLDT_Float: return DataTypeToCudnn<vl::VLDT_Float>::dataType ;
      case VLDT_Double: return DataTypeToCudnn<vl::VLDT_Double>::dataType ;
      default: assert(false) ; return CUDNN_DATA_FLOAT ; // bogus
    }
  }

  // -------------------------------------------------------------------
  /// MARK: - Defer
  // ----------------------------------------------------------------

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

  // -------------------------------------------------------------------
  /// MARK: - CudnnTensorDescripor
  // -------------------------------------------------------------------

  class CudnnTensorDescriptor
  {
  public:
    CudnnTensorDescriptor() : desc(0), initialized(false) { }
    ~CudnnTensorDescriptor() ;
    cudnnStatus_t init(DataType dataType, TensorShape const & shape) ;
    cudnnTensorDescriptor_t get() const ;
    void clear() ;
    operator bool() { return initialized ; }
  private:
    cudnnTensorDescriptor_t desc ;
    bool initialized ;
  } ;

  inline CudnnTensorDescriptor::~CudnnTensorDescriptor()
  {
    clear() ;
  }

  inline void CudnnTensorDescriptor::clear()
  {
    if (initialized) {
      cudnnDestroyTensorDescriptor(desc) ;
      desc = NULL ;
      initialized = false ;
    }
  }

  inline cudnnStatus_t
  CudnnTensorDescriptor::init(DataType dataType, TensorShape const & shape)
  {
    cudnnDataType_t dt ;
    switch (dataType) {
      case VLDT_Float: dt = DataTypeToCudnn<VLDT_Float>::dataType ; break ;
      case VLDT_Double: dt = DataTypeToCudnn<VLDT_Double>::dataType ; break ;
      default: assert(false) ;
    }

    auto status = cudnnCreateTensorDescriptor(&desc) ;
    if (status != CUDNN_STATUS_SUCCESS) {
      return status ;
    }
    initialized = true ;

    status = cudnnSetTensor4dDescriptorEx
    (desc, dt,
     (int)shape.getCardinality(),
     (int)shape.getNumChannels(),
     (int)shape.getWidth(),
     (int)shape.getHeight(),
     (int)(shape.getHeight()*shape.getWidth()*shape.getNumChannels()), //strides
     (int)(shape.getHeight()*shape.getWidth()),
     (int)shape.getHeight(),
     1) ;

    return status ;
  }

  inline cudnnTensorDescriptor_t
  CudnnTensorDescriptor::get() const
  {
    return desc ;
  }

} }

#endif /* cudnnhelper_h */
