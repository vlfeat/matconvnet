// @file data.hpp
// @brief Basic data structures
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl_data_hpp__
#define __vl_data_hpp__

#include <cstddef>
#include <string>
#include <vector>
#include <sstream>

#include "impl/compat.h"

#define STRINGIZE(x) STRINGIZE_HELPER(x)
#define STRINGIZE_HELPER(x) #x
#define FILELINE STRINGIZE(__FILE__) ":" STRINGIZE(__LINE__)
#define divides(a,b) ((b) == (b)/(a)*(a))

#if ENABLE_DOUBLE
#define IF_DOUBLE(x) x
#else
#define IF_DOUBLE(x)
#endif

#define VL_M_PI 3.14159265358979323846
#define VL_M_PI_F 3.14159265358979323846f

#define VLERR(code,message) \
getContext().passError(code,message)

#define VLLOG(op,level) \
if ((op).getContext().getLogLevel() < level) { } \
else vl::Context::Logger((op).getContext()).getStream()

namespace vl {

  /// Basic integral type.
  typedef ptrdiff_t Int ;
  
  /// Error codes.
  enum ErrorCode {
    VLE_Success = 0,
    VLE_Unsupported,
    VLE_Cuda,
    VLE_Cudnn,
    VLE_Cublas,
    VLE_OutOfMemory,
    VLE_OutOfGPUMemeory,
    VLE_IllegalArgument,
    VLE_Timeout,
    VLE_NoData,
    VLE_IllegalMessage,
    VLE_Interrupted,
    VLE_TensorShapeMismatch,
    VLE_TensorTypeMismatch,
    VLE_Unknown,
  } ;

  /// Get an error message for a given code.
  const char * getErrorMessage(ErrorCode error) ;

  /// Type of device: CPU or GPU.
  enum DeviceType {
    VLDT_CPU = 0,
    VLDT_GPU
  }  ;

  /// Type of data (char, float, double, ...).
  enum DataType {
    VLDT_Char,
    VLDT_Float,
    VLDT_Double
  } ;

  /// Convert unsigned data to signed.
  template<typename T>
  static inline typename std::make_signed<T>::type as_signed(T x)
  {
    return static_cast<typename std::make_signed<T>::type >(x) ;
  }

  /// Convert unsigned data to unsigned.
  template<typename T>
  static inline typename std::make_unsigned<T>::type as_unsigned(T x)
  {
    return static_cast<typename std::make_unsigned<T>::type >(x) ;
  }

  /// Convert a DataType code to a C++ type.
  template <vl::DataType dataType> struct DataTypeTraits { } ;
  template <> struct DataTypeTraits<VLDT_Char> {
    typedef char type ;
    static constexpr std::size_t size = sizeof(char) ;
    static constexpr auto name = "char" ;
  } ;
  template <> struct DataTypeTraits<VLDT_Float> {
    typedef float type ;
    static constexpr std::size_t size = sizeof(float) ;
    static constexpr auto name = "float" ;
  } ;
  template <> struct DataTypeTraits<VLDT_Double> {
    typedef double type ;
    static constexpr std::size_t size = sizeof(double) ;
    static constexpr auto name = "double" ;
  } ;


  /// Convert a DataType code to a C++ type.
  template <vl::DeviceType deviceType> struct DeviceTypeTraits { } ;
  template <> struct DeviceTypeTraits<VLDT_CPU> {
    static constexpr auto name = "CPU" ;
  } ;
  template <> struct DeviceTypeTraits<VLDT_GPU> {
    static constexpr auto name = "GPU" ;
  } ;

  /// Convert a C++ type to a DataType code.
  template <typename type> struct BuiltinToDataType {} ;
  template <> struct BuiltinToDataType<char> { enum { dataType = VLDT_Char } ; } ;
  template <> struct BuiltinToDataType<float> { enum { dataType = VLDT_Float } ; } ;
  template <> struct BuiltinToDataType<double> { enum { dataType = VLDT_Double } ; } ;

  /// Get the size of a datat type in bytes.
  inline size_t getDataTypeSizeInBytes(DataType dataType) {
    switch (dataType) {
      case VLDT_Char:   return DataTypeTraits<VLDT_Char>::size ;
      case VLDT_Float:  return DataTypeTraits<VLDT_Float>::size ;
      case VLDT_Double: return DataTypeTraits<VLDT_Double>::size ;
      default:          return 0 ;
    }
  }

  class CudaHelper ;

  /* -----------------------------------------------------------------
   * Helpers
   * -------------------------------------------------------------- */

  /// Computes the smallest multiple of @a b which is greater
  /// or equal to @a a.
  template<typename type>
  inline type divideAndRoundUp(type a, type b)
  {
    return (a + b - 1) / b ;
  }

  /// Compute the greatest common divisor g of non-negative integers
  /// @a a and @a b as well as two integers @a u and @a v such that
  /// $au + bv = g$ (Bezout's coefficients).
  Int gcd(Int a, Int b, Int &u, Int& v) ;

  /// Draw a Normally-distributed scalar.
  double randn() ;

  /// Get the value of the realtime clock in microseconds.
  size_t getTime() ;

  namespace impl {
    class Buffer
    {
    public:
      Buffer() ;
      vl::ErrorCode init(DeviceType deviceType, DataType dataType, size_t size) ;
      void * getMemory() ;
      int getNumReallocations() const ;
      void clear() ;
      void invalidateGpu() ;
    protected:
      DeviceType deviceType ;
      DataType dataType ;
      size_t size ;
      void * memory ;
      int numReallocations ;
    } ;
  }

  template<class T>
  class pretty_helper
  {
  public:
    pretty_helper(T const& object) : object(object) { }
    T const& get() const { return object ; }
  private:
    T const& object ;
  } ;

  template<class T>
  pretty_helper<T> pretty(T const & vec) {
    return {vec} ;
  }

  template<class T>
  std::ostream & operator << (std::ostream& os, pretty_helper<std::vector<T>> const& x) {
    auto const& vec = x.get() ;
    os << "[" ;
    for (auto i = begin(vec) ; i != end(vec) ; ++i) {
      os << *i ;
      if (i + 1 != end(vec)) os << " " ;
    }
    return os << "]" ;
  }
  
  /* -----------------------------------------------------------------
   * Context
   * -------------------------------------------------------------- */

  class Context
  {
  public:
    Context() ;
    ~Context() ;

    void * getWorkspace(DeviceType device, size_t size) ;
    void clearWorkspace(DeviceType device) ;
    void * getAllOnes(DeviceType device, DataType type, size_t size) ;
    void clearAllOnes(DeviceType device) ;
    CudaHelper& getCudaHelper() ;

    void clear() ; // do a reset
    void invalidateGpu() ; // drop CUDA memory and handles

    void setLogLevel(int level) ;
    int getLogLevel() const ;
    void log(std::string const& message) ;
    void clearLog() ;
    std::string const& getLastLoggedMessage() const ;
    std::vector<std::string> const& getLogbook() const ;

    class Logger {
    public:
      Logger(Context &context) : context(context) { }
      ~Logger() { context.log(logStream.str()) ; }
      std::ostringstream& getStream() { return logStream ; }
    private:
      Logger(Logger &x) = delete ;
      Logger(Logger &&x) = delete ;
      Logger& operator=(Logger&) = delete ;
      Context& context ;
      std::ostringstream logStream ;
    } ;

    vl::ErrorCode passError(vl::ErrorCode error, char const * message = NULL) ;
    vl::ErrorCode setError(vl::ErrorCode error, char const * message = NULL) ;
    void resetLastError() ;
    vl::ErrorCode getLastError() const ;
    std::string const& getLastErrorMessage() const ;

  private:
    impl::Buffer workspace[2] ;
    impl::Buffer allOnes[2] ;

    ErrorCode lastError ;
    std::string lastErrorMessage ;
    std::vector<std::string> logbook ;
    int logLevel ;

    CudaHelper * cudaHelper ;
  } ;

  /* -----------------------------------------------------------------
   * TensorShape
   * -------------------------------------------------------------- */

  class TensorShape
  {
  public:
    static constexpr Int maxNumDimensions = 8 ;

    TensorShape() ;
    TensorShape(TensorShape const& t) ;
    TensorShape(const std::initializer_list<Int> &dims) ;
    TensorShape(const std::vector<Int> &dims) ;
    TensorShape(Int height, Int width, Int depth, Int size) ;
    TensorShape(Int const * dimensions, Int numDimensions) ;

    void clear() ; // set to empty (numDimensions = 0)
    void setDimension(Int num, Int dimension) ;
    void setDimensions(Int const * dimensions, Int numDimensions) ;
    void setHeight(Int x) ;
    void setWidth(Int x) ;
    void setDepth(Int x) ;
    void setSize(Int x) ;
    void reshape(Int numDimensions) ; // squash or stretch to numDimensions
    void reshape(TensorShape const & shape) ; // same as operator=

    Int getDimension(Int num) const ;
    Int const * getDimensionsAsPtr() const ;
    Int getNumDimensions() const ;
    std::vector<Int> getDimensions() const ;
    Int getHeight() const ;
    Int getWidth() const ;
    Int getNumChannels() const ;
    Int getCardinality() const ;

    Int getNumElements() const ;
    bool isEmpty() const ;

  protected:
    Int dimensions [maxNumDimensions] ;
    Int numDimensions ;
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
    Tensor(TensorShape const & shape, DataType dataType,
           DeviceType deviceType, void * memory, size_t memorySize) ;
    void * getMemory() ;
    void const * getMemory() const ;
    DeviceType getDeviceType() const ;
    TensorShape getShape() const ;
    DataType getDataType() const ;
    operator bool() const ;
    bool isNull() const ;
    void setMemory(void * x) ;

  protected:
    DeviceType deviceType ;
    DataType dataType ;
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
