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
        base.getContext().setError
        (base.getContext().getCudaHelper().catchCudaError("GPU")) ;
      }
      return base.getContext().passError(error, "Dispatcher") ;
    }
#endif
    switch (tt.dataType) {
      case vl::VLDT_Float:
        error = C<vl::VLDT_CPU,vl::VLDT_Float>()(base,args...) ;
        break ;
#if ENABLE_DOUBLE
      case vl::VLDT_Double:
        error = C<vl::VLDT_CPU,vl::VLDT_Double>()(base,args...) ;
        break ;
#endif
      default: assert(false) ;
    }
    return base.getContext().passError(error, "Dispatcher") ;
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
    if (!base.getContext().getCudaHelper().getCudnnEnabled()) goto fallback;
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
      return base.getContext().passError(error, "DispatcherCUDNN") ;
    }
  fallback:
#endif
    return dispatch<C>()(base,args...) ;
  }
} ;

// -------------------------------------------------------------------
/// MARK: - Check tensor compatibility
// -------------------------------------------------------------------

// Check that all the arguments are tensor of the same data and device
// type (e.g. all FLOAT GPU tensors). Skip any tensor which is
// either empty or null (forgotten) as these denote missing arguments
// or only store shape information.
static inline
std::tuple<bool, bool, vl::DeviceType, vl::DataType>
check_tensor_compatibility_impl() {
  return {true,false,VLDT_CPU,VLDT_Float} ; // CPU/Float are placeholder values
}

template <typename ... Types>
static inline
std::tuple<bool, bool, vl::DeviceType, vl::DataType>
check_tensor_compatibility_impl(vl::Tensor const &t, Types& ... args) {
  auto result = check_tensor_compatibility_impl(args...) ;
  if (t.isEmpty() || t.isNull()) {
    return result ;
  } else {
    if (!std::get<1>(result)) {
      // No non-empty tensor found so far.
      return {true,true,t.getDeviceType(),t.getDataType()} ;
    } else {
      // There was a non-empty tensor before.
      bool compatible = std::get<0>(result)
      && (t.getDeviceType() == std::get<2>(result))
      && (t.getDataType() == std::get<3>(result)) ;
      return {compatible,true,std::get<2>(result),std::get<3>(result)} ;
    }
  }
}

template <typename ... Types>
static inline
bool check_tensor_compatibility(Types& ... args) {
  auto result = check_tensor_compatibility_impl(args...) ;
  return std::get<0>(result) ;
}

} }

#endif
