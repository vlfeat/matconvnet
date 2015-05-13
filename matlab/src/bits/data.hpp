// @file data.hpp
// @brief Basic data structures
// @author Andrea Vedaldi

/*
Copyright (C) 2015 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl_data_hpp__
#define __vl_data_hpp__

#include <cstddef>
#include <string>

#define STRINGIZE(x) STRINGIZE_HELPER(x)
#define STRINGIZE_HELPER(x) #x
#define FILELINE STRINGIZE(__FILE__) ":" STRINGIZE(__LINE__)

namespace vl {
  typedef int index_t ;
  enum Device { CPU = 0, GPU }  ;
  enum Type {
    vlTypeChar,
    vlTypeFloat,
    vlTypeDouble
  } ;

  enum Error {
    vlSuccess = 0,
    vlErrorUnsupported,
    vlErrorCuda,
    vlErrorCudnn,
    vlErrorCublas,
    vlErrorOutOfMemory,
    vlErrorOutOfGPUMemeory,
    vlErrorUnknown
  } ;
  const char * getErrorMessage(Error error) ;

  class CudaHelper ;

  /* -----------------------------------------------------------------
   * Helpers
   * -------------------------------------------------------------- */

  inline int divideUpwards(int a, int b)
  {
    return (a + b - 1) / b ;
  }

  namespace impl {
    class Buffer
    {
    public:
      Buffer() ;
      vl::Error init(Device deviceType, Type dataType, size_t size) ;
      void * getMemory() ;
      int getNumReallocations() const ;
      void clear() ;
      void invalidateGpu() ;
    protected:
      Device deviceType ;
      Type dataType ;
      size_t size ;
      void * memory ;
      int numReallocations ;
    } ;
  }

  /* -----------------------------------------------------------------
   * Context
   * -------------------------------------------------------------- */

  class Context
  {
  public:
    Context() ;
    ~Context() ;

    void * getWorkspace(Device device, size_t size) ;
    void clearWorkspace(Device device) ;
    void * getAllOnes(Device device, Type type, size_t size) ;
    void clearAllOnes(Device device) ;
    CudaHelper& getCudaHelper() ;

    void clear() ; // do a reset
    void invalidateGpu() ; // drop CUDA memory and handles

    vl::Error passError(vl::Error error, char const * message = NULL) ;
    vl::Error setError(vl::Error error, char const * message = NULL) ;
    void resetLastError() ;
    vl::Error getLastError() const ;
    std::string const& getLastErrorMessage() const ;

  private:
    impl::Buffer workspace[2] ;
    impl::Buffer allOnes[2] ;

    Error lastError ;
    std::string lastErrorMessage ;

    CudaHelper * cudaHelper ;
  } ;

  /* -----------------------------------------------------------------
   * TensorGeometry
   * -------------------------------------------------------------- */

  class TensorGeometry
  {
  public:
    TensorGeometry() ;
    TensorGeometry(TensorGeometry const& t) ;
    TensorGeometry(index_t height, index_t width, index_t depth, index_t size) ;
    index_t getHeight() const ;
    index_t getWidth() const ;
    index_t getDepth() const ;
    index_t getSize() const ;
    index_t getNumElements() const ;
    bool isEmpty() const ;
    void setHeight(index_t x) ;
    void setWidth(index_t x) ;
    void setDepth(index_t x) ;
    void setSize(index_t x) ;

  protected:
    index_t height ;
    index_t width ;
    index_t depth ;
    index_t size ;
  } ;

  inline TensorGeometry::TensorGeometry(TensorGeometry const & t)
  : height(t.height), width(t.width), depth(t.depth), size(t.size)
  { }

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
    Tensor(Tensor const &) ;
    Tensor(float * memory, size_t memorySize, Device memoryType,
           TensorGeometry const & geom) ;
    float * getMemory() ;
    Device getMemoryType() const ;
    TensorGeometry getGeometry() const ;
    operator bool() const ;
    bool isNull() const ;
    void setMemory(float * x) ;

  protected:
    float * memory ;
    size_t memorySize ;
    Device memoryType ;
  } ;

  inline Tensor::Tensor(Tensor const& t)
  : TensorGeometry(t), memory(t.memory), memorySize(t.memorySize), memoryType(t.memoryType)
  { }

  inline bool areCompatible(Tensor const & a, Tensor const & b)
  {
    return
    (a.isEmpty() || a.isNull()) ||
    (b.isEmpty() || b.isNull()) ||
    (a.getMemoryType() == b.getMemoryType()) ;
  }
}

#endif
