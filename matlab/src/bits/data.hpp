// @file data.hpp
// @brief Basic data structures
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__data_hpp__
#define __vl__data_hpp__

#include <cstddef>
#include <string>

#define STRINGIZE(x) STRINGIZE_HELPER(x)
#define STRINGIZE_HELPER(x) #x
#define FILELINE STRINGIZE(__FILE__) ":" STRINGIZE(__LINE__)
#define divides(a,b) ((b) == (b)/(a)*(a))

#ifdef _MSC_VER
#define __func__ __FUNCTION__
#endif

#if ENABLE_DOUBLE
#define IF_DOUBLE(x) x
#else
#define IF_DOUBLE(x)
#endif

namespace vl {
  enum Device { CPU = 0, GPU }  ;
  enum Type {
    vlTypeChar,
    vlTypeFloat,
    vlTypeDouble
  } ;

  template <vl::Type id> struct DataTypeTraits { } ;
  template <> struct DataTypeTraits<vlTypeChar> { typedef char type ; } ;
  template <> struct DataTypeTraits<vlTypeFloat> { typedef float type ; } ;
  template <> struct DataTypeTraits<vlTypeDouble> { typedef double type ; } ;

  template <typename type> struct BuiltinToDataType {} ;
  template <> struct BuiltinToDataType<char> { enum { dataType = vlTypeChar } ; } ;
  template <> struct BuiltinToDataType<float> { enum { dataType = vlTypeFloat } ; } ;
  template <> struct BuiltinToDataType<double> { enum { dataType = vlTypeDouble } ; } ;

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
   * TensorShape
   * -------------------------------------------------------------- */

#define VL_TENSOR_SHAPE_MAX_NUM_DIMENSIONS 8

  class TensorShape
  {
  public:
    TensorShape() ;
    TensorShape(TensorShape const& t) ;
    TensorShape(size_t height, size_t width, size_t depth, size_t size) ;
    TensorShape(size_t const * dimensions, size_t numDimensions) ;

    void clear() ; // set to empty (numDimensions = 0)
    void setDimension(size_t num, size_t dimension) ;
    void setDimensions(size_t const * dimensions, size_t numDimensions) ;
    void setHeight(size_t x) ;
    void setWidth(size_t x) ;
    void setDepth(size_t x) ;
    void setSize(size_t x) ;
    void reshape(size_t numDimensions) ; // squash or stretch to numDimensions
    void reshape(TensorShape const & shape) ; // same as operator=

    size_t getDimension(size_t num) const ;
    size_t const * getDimensions() const ;
    size_t getNumDimensions() const ;
    size_t getHeight() const ;
    size_t getWidth() const ;
    size_t getDepth() const ;
    size_t getSize() const ;

    size_t getNumElements() const ;
    bool isEmpty() const ;

  protected:
    size_t dimensions [VL_TENSOR_SHAPE_MAX_NUM_DIMENSIONS] ;
    size_t numDimensions ;
  } ;

  bool operator == (TensorShape const & a, TensorShape const & b) ;

  inline bool operator != (TensorShape const & a, TensorShape const & b)
  {
    return ! (a == b) ;
  }

  /* -----------------------------------------------------------------
   * Tensor
   * -------------------------------------------------------------- */

  class Tensor : public TensorShape
  {
  public:
    Tensor() ;
    Tensor(Tensor const &) ;
    Tensor(TensorShape const & shape, Type dataType,
           Device deviceType, void * memory, size_t memorySize) ;
    void * getMemory() ;
    Device getDeviceType() const ;
    TensorShape getShape() const ;
    Type getDataType() const ;
    operator bool() const ;
    bool isNull() const ;
    void setMemory(void * x) ;

  protected:
    Device deviceType ;
    Type dataType ;
    void * memory ;
    size_t memorySize ;
  } ;

  inline Tensor::Tensor(Tensor const& t)
  : TensorShape(t), dataType(t.dataType), deviceType(t.deviceType),
  memory(t.memory), memorySize(t.memorySize)
  { }

  inline bool areCompatible(Tensor const & a, Tensor const & b)
  {
    return
    (a.isEmpty() || a.isNull()) ||
    (b.isEmpty() || b.isNull()) ||
    ((a.getDeviceType() == b.getDeviceType()) & (a.getDataType() == b.getDataType())) ;
  }
}

#endif
