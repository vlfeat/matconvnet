// @file dispatcher.hpp
// @brief Dispatcher helper function matconvnet
// @author Andrea Vedaldi

/*
Copyright (C) 2017 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __dispatcher_hpp__
#define __dispatcher_hpp__

#include "../data.hpp"
#include <cassert>

namespace vl { namespace impl {

// -------------------------------------------------------------------
//                                                             Helpers
// -------------------------------------------------------------------
// A function to check variadic template argument for tensors and
// extract their types. This is used to dispatch code to CPU/GPU
// and float/double.
//
// The last non-null tensor is used to extract this information.

struct holder {
  vl::Tensor *tensor ;
  holder(vl::Tensor & t) : tensor(&t) { }
  template<typename type> holder(type &t) : tensor(NULL) { }
} ;

struct tensor_type {
  vl::DeviceType deviceType = VLDT_CPU ;
  vl::DataType dataType = VLDT_Float ;
} ;

template <typename ... Types>
tensor_type findTensorType(Types& ... args)
{
  holder holders [] = {args...} ;
  tensor_type tt ;
  for (auto & h : holders) {
    if (h.tensor && !h.tensor->isNull()) {
      tt.deviceType = h.tensor->getDeviceType() ;
      tt.dataType = h.tensor->getDataType() ;
    }
  }
  return tt ;
}

// -------------------------------------------------------------------
//                                                            Dispatch
// -------------------------------------------------------------------

template <template <vl::DeviceType deviceType, vl::DataType dataType> class C>
struct dispatch
{
  template <class B, typename ... Types>
  vl::ErrorCode operator()(B& base, Types& ... args)
  {
    vl::ErrorCode error ;
    tensor_type tt = findTensorType(args...) ;
#if ENABLE_GPU
    if (tt.deviceType == VLDT_GPU) {
      switch (tt.dataType) {
        case vl::VLDT_Float:
          error = C<vl::VLDT_GPU,vl::VLDT_Float>()(base,args...) ;
          break ;
#if ENABLE_DOUBLE
        case vl::VLDT_Double:
          error = C<vl::VLDT_GPU,vl::VLDT_Double>()(base,args...) ;
          break ;
#endif
        default: assert(false) ;
      }
      if (error == vl::VLE_Cuda) {
        base.context.setError
        (base.context.getCudaHelper().catchCudaError("GPU")) ;
      }
      return base.context.passError(error, __func__) ;
    }
#endif
    switch (tt.dataType) {
      case vl::VLDT_Float:
        error = C<vl::VLDT_CPU,vl::VLDT_Float>()(base,args...) ;
        break ;
      case vl::VLDT_Double:
        error = C<vl::VLDT_CPU,vl::VLDT_Double>()(base,args...) ;
        break ;
      default: assert(false) ;
    }
    return base.context.passError(error, __func__) ;
  }
} ;

template <
template <vl::DeviceType deviceType, vl::DataType dataType> class C,
template <vl::DataType dataType> class CU >
struct dispatch_cudnn
{
  template <class B, typename ... Types>
  vl::ErrorCode operator()(B& base, Types& ... args)
  {
#if ENABLE_CUDNN
    tensor_type tt = findTensorType(args...) ;

    vl::ErrorCode error ;
    if (tt.deviceType == vl::VLDT_GPU) {
      switch (tt.dataType) {
        case vl::VLDT_Float:
          error = CU<vl::VLDT_Float>()(base,args...) ;
          break ;
#if ENABLE_DOUBLE
        case vl::VLDT_Double:
          error = CU<vl::VLDT_Double>()(base,args...) ;
          break ;
#endif
        default: assert(false) ;
      }
      if (error == vl::VLE_Success) { return error ; }
      if (error == vl::VLE_Unsupported) { goto fallback ; }
      return base.context.passError(error, __func__) ;
    }
  fallback:
#endif
    return dispatch<C>()(base,args...) ;
  }
} ;

} }

#endif
