//
//  datamex.h
//  matconv
//
//  Created by Andrea Vedaldi on 31/01/2015.
//  Copyright (c) 2015 Andrea Vedaldi. All rights reserved.
//

#ifndef __matconv__datamex__
#define __matconv__datamex__

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

  void print(char const * str, TensorGeometry const & tensor) ;
}


#endif /* defined(__matconv__datamex__) */
