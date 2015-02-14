// @file datamex.hpp
// @brief Basic data structures (MEX support)
// @author Andrea Vedaldi

/*
Copyright (C) 2015 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__datamex__
#define __vl__datamex__

#include "mex.h"

#ifdef ENABLE_GPU
#include "gpu/mxGPUArray.h"
#endif

#include "data.hpp"

namespace vl {

  class MexTensor : public Tensor
  {
  public:
    MexTensor() ;
    MexTensor(Device type, TensorGeometry const & geom) ;
    MexTensor(Device type, TensorGeometry const & geom, float value) ;
    MexTensor(mxArray const * array) ;
    ~MexTensor() ;
    mxArray * relinquish() ;
    void clear() ;
    MexTensor & operator= (MexTensor & tensor) ;

    // Allow copying an rvalue MexTensor (similar to auto_ptr)
    struct MexTensorRef {
      inline explicit MexTensorRef(MexTensor& tensor_) : tensor(tensor_) { }
      MexTensor & tensor ;
    } ;
    inline operator MexTensorRef() {
      return MexTensorRef(*this) ;
    }
    inline MexTensor & operator= (MexTensorRef ref) {
      *this = ref.tensor ;
      return *this ;
    }

  protected:
    mxArray const * array ;
#ifdef ENABLE_GPU
    mxGPUArray const * gpuArray ;
#endif
    bool isArrayOwner ;
    void allocInitialized() ;
    void allocUninitialized() ;

  private: // prevention
    MexTensor(MexTensor const &) ;
  } ;

  void print(char const * str, Tensor const & tensor) ;

  void mexThrowError(Context const& context, vl::Error error) ;
}


#endif /* defined(__vl__datamex__) */
