//
//  data.hpp
//  matconv
//
//  Created by Andrea Vedaldi on 30/01/2015.
//  Copyright (c) 2015 Andrea Vedaldi. All rights reserved.
//

#ifndef __matconvnet_data_hpp__
#define __matconvnet_data_hpp__

#include <cstddef>

namespace vl {

  typedef int index_t ;
  enum Device { CPU, GPU }  ;
  enum Type { FLOAT, DOUBLE } ;
  enum Status { SUCCESS = 0, UNSUPPORTED = 1, ERROR = 2 } ;
  class CudaHelper ;

  /* -----------------------------------------------------------------
   * Helpers
   * -------------------------------------------------------------- */

  inline int divideUpwards(int a, int b)
  {
    return (a + b - 1) / b ;
  }

  /* -----------------------------------------------------------------
   * Context
   * -------------------------------------------------------------- */

  class Context
  {
  public:
    Context() ;
    ~Context() ;

    void reset() ;
    void * getWorkspace(Device device, size_t size) ;
    void clearWorkspace(Device device) ;
    void * getAllOnes(Device device, Type type, size_t size) ;
    void clearAllOnes(Device device) ;
    CudaHelper& getCudaHelper() ;

  private:
    void * cpuWorkspace ;
    size_t cpuWorkspaceSize ;

    void * cpuAllOnes ;
    size_t cpuAllOnesSize ;
    Type cpuAllOnesType ;

    void * gpuWorkspace ;
    size_t gpuWorkspaceSize ;

    void * gpuAllOnes ;
    size_t gpuAllOnesSize ;
    Type gpuAllOnesType ;

    CudaHelper * cudaHelper ;
  } ;

  /* -----------------------------------------------------------------
   * TensorGeometry
   * -------------------------------------------------------------- */

  class TensorGeometry
  {
  public:
    TensorGeometry() ;
    TensorGeometry(index_t height, index_t width, index_t depth, index_t size) ;
    index_t getHeight() const ;
    index_t getWidth() const ;
    index_t getDepth() const ;
    index_t getSize() const ;
    index_t getNumElements() const ;
    bool isEmpty() const ;

  protected:
    index_t height ;
    index_t width ;
    index_t depth ;
    index_t size ;
  } ;

  inline bool operator == (TensorGeometry const & a, TensorGeometry const & b)
  {
    return
    (a.getHeight() == b.getHeight()) &
    (a.getWidth() == b.getWidth()) &
    (a.getDepth() == b.getDepth()) &
    (a.getSize() == b.getSize()) ;
  }

  inline bool operator != (TensorGeometry const & a, TensorGeometry const & b)
  {
    return ! (a == b) ;
  }

  /* -----------------------------------------------------------------
   * Tensor
   * -------------------------------------------------------------- */

  class Tensor : public TensorGeometry
  {
  public:
    Tensor() ;
    Tensor(float * memory, size_t memorySize, Device memoryType,
           TensorGeometry const & geom) ;
    float * getMemory() ;
    Device getMemoryType() const ;
    TensorGeometry getGeometry() const ;
    operator bool() const ;
    bool isNull() const ;

  protected:
    float * memory ;
    size_t memorySize ;
    Device memoryType ;
  } ;

  inline bool areCompatible(Tensor const & a, Tensor const & b)
  {
    return
    (a.isEmpty() || a.isNull()) ||
    (b.isEmpty() || b.isNull()) ||
    (a.getMemoryType() == b.getMemoryType()) ;
  }
}

#endif
