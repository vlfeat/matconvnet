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

// -------------------------------------------------------------------
//                                                            Dispatch
// -------------------------------------------------------------------

template <template <vl::DeviceType deviceType, vl::DataType dataType> class C>
struct dispatch
{
  template <class B, typename ... Types>
  vl::ErrorCode operator()(B& base, vl::Tensor output, Types ... args)
  {
    vl::ErrorCode error ;
#if ENABLE_GPU
    if (output.getDeviceType() == vl::VLDT_GPU) {
      switch (output.getDataType()) {
        case vl::VLDT_Float:
          error = C<vl::VLDT_GPU,vl::VLDT_Float>()(base,output,args...) ;
          break ;
        case vl::VLDT_Double:
          error = C<vl::VLDT_GPU,vl::VLDT_Double>()(base,output,args...) ;
          break ;
        default: assert(false) ;
      }
      if (error == vl::VLE_Cuda) {
        base.context.setError
        (base.context.getCudaHelper().catchCudaError("GPU")) ;
      }
      return base.context.passError(error, __func__) ;
    }
#endif
    switch (output.getDataType()) {
      case vl::VLDT_Float:
        error = C<vl::VLDT_CPU,vl::VLDT_Float>()(base,output,args...) ;
        break ;
      case vl::VLDT_Double:
        error = C<vl::VLDT_CPU,vl::VLDT_Double>()(base,output,args...) ;
        break ;
      default: assert(false) ;
    }
    return base.context.passError(error, __func__) ;
  }
} ;

#endif
