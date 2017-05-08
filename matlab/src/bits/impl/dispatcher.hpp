//
//  dispatcher.hpp
//  matconvnet
//
//  Created by Andrea Vedaldi on 08/05/2017.
//  Copyright © 2017 Andrea Vedaldi. All rights reserved.
//

#ifndef __dispatcher_hpp__
#define __dispatcher_hpp__

#include "../data.hpp"
#include <cassert>

// -------------------------------------------------------------------
//                                                          Dispatcher
// -------------------------------------------------------------------

template < template <vl::DeviceType deviceType, vl::DataType dataType> class C>
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
      return error ;
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
    return error ;
  }
} ;

#endif
