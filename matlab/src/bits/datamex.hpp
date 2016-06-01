// @file datamex.hpp
// @brief Basic data structures (MEX support)
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__datamex__
#define __vl__datamex__

#include "mex.h"

#if ENABLE_GPU
#include "gpu/mxGPUArray.h"
#endif

#include "data.hpp"

namespace vl {

  class MexTensor ;

  class MexContext : public Context
  {
  public:
    MexContext() ;
    ~MexContext() ;

  protected:
#if ENABLE_GPU
    vl::Error initGpu() ;
    vl::Error validateGpu() ;
    mxArray * canary ; // if it breathes, the GPU state is valid
    bool gpuIsInitialized ;
#endif

    friend class MexTensor ;
  } ;

  class MexTensor : public Tensor
  {
  public:
    MexTensor(MexContext & context) ;
    vl::Error init(mxArray const * array) ;
    vl::Error init(Device deviceType, Type dataType, TensorShape const & shape) ;
    vl::Error initWithZeros(Device deviceType, Type dataType, TensorShape const & shape) ;
    vl::Error initWithValue(Device deviceType, Type dataType, TensorShape const & shape, double value) ;

    mxArray * relinquish() ;
    void clear() ;
    ~MexTensor() ;

    size_t getMemorySize() const ;

  protected:
    MexContext & context ;
    mxArray const * array ;
#ifdef ENABLE_GPU
    mxGPUArray const * gpuArray ;
#endif
    bool isArrayOwner ;

  private: // prevention
    MexTensor(MexTensor const &) ;
    MexTensor & operator= (MexTensor & tensor) ;
    vl::Error initHelper(Device deviceType, Type dataType, TensorShape const & shape, bool fillWithZeros = false) ;
  } ;

  void print(char const * str, MexTensor const & tensor) ;

  void mexThrowError(Context const& context, vl::Error error) ;
}


#endif /* defined(__vl__datamex__) */
