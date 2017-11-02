// @file datamex.hpp
// @brief Basic data structures (MEX support)
// @author Andrea Vedaldi

/*
Copyright (C) 2015-17 Andrea Vedaldi.
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

#if ENABLE_CUDNN
#include "cudnn.h"
#endif

#include "data.hpp"
#include "mexutils.h"

#define MXCHECK(x) \
{ vl::ErrorCode error = (x) ; if (error != vl::VLE_Success) { return error ; } }

#define MXOPTVEC(x,y) \
{ std::vector<Int> x ; \
if (context.parse(x,optarg) != vl::VLE_Success) { \
return context.passError(vl::VLE_IllegalArgument, "Could not set " #x ":") ; } \
vl::ErrorCode error = (op.y(x)) ; \
if (error != vl::VLE_Success) { return error ; } }

namespace vl {

  class MexTensor ;

  class MexContext : public Context
  {
  public:
    MexContext() ;
    ~MexContext() ;

    ErrorCode parse(std::vector<Int>& vec, mxArray const* array) {
      if (!vlmxIsPlainMatrix(array,-1,-1)) {
        return setError(vl::VLE_IllegalArgument, "Not a plain vector.") ;
      }
      vec = std::move(std::vector<Int>{
        mxGetPr(array), mxGetPr(array) + mxGetNumberOfElements(array)}) ;
      return VLE_Success ;
    }

  protected:
#if ENABLE_GPU
    vl::ErrorCode initGpu() ;
    vl::ErrorCode validateGpu() ;
    mxArray * canary ; // if it breathes, the GPU state is valid
    bool gpuIsInitialized ;
#endif

    friend class MexTensor ;
  } ;

  class MexTensor : public Tensor
  {
  public:
    MexTensor(MexContext & context) ;
    MexTensor(MexTensor const &) = delete ;
    MexTensor & operator= (MexTensor & tensor) = delete ;

    vl::ErrorCode init(mxArray const * array) ;
    vl::ErrorCode init(DeviceType deviceType, DataType dataType, TensorShape const & shape) ;
    vl::ErrorCode initWithZeros(DeviceType deviceType, DataType dataType, TensorShape const & shape) ;
    vl::ErrorCode initWithValue(DeviceType deviceType, DataType dataType, TensorShape const & shape, double value) ;

    void makePersistent() ;
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

  private:
    vl::ErrorCode initHelper(DeviceType deviceType, DataType dataType, TensorShape const & shape, bool fillWithZeros = false) ;
  } ;

  void print(char const * str, MexTensor const & tensor) ;
  void mexThrowError(Context const& context, vl::ErrorCode error) ;
}

#endif /* defined(__vl__datamex__) */
