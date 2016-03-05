// @file datamex.cu
// @brief Basic data structures (MEX support)
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "datamex.hpp"
#if ENABLE_GPU
#include "datacu.hpp"
#endif

#ifndef NDEBUG
#include<iostream>
#endif

#include "impl/copy.hpp"

using namespace vl ;
using namespace vl::impl ;

/* ---------------------------------------------------------------- */
/*                                                       MexContext */
/* ---------------------------------------------------------------- */

vl::MexContext::MexContext()
: Context()
#if ENABLE_GPU
, gpuIsInitialized(false)
, canary(NULL)
#endif
{ }

vl::MexContext::~MexContext()
{
#if ENABLE_GPU
  // so that ~Context does not crash if MATLAB resets the GPU in the mean time
  validateGpu() ;
#endif
}

/* ---------------------------------------------------------------- */
/*                                                   GPU management */
/* ---------------------------------------------------------------- */

#if ENABLE_GPU

// Do noting if the GPU is not initialized, otherwise invalidate it
// if needed
vl::Error
MexContext::validateGpu()
{
  if (!gpuIsInitialized) { return vl::vlSuccess ; }
  gpuIsInitialized = mxGPUIsValidGPUData(canary) ;
  if (!gpuIsInitialized) {
#ifndef NDEBUG
    std::cout<<"MexContext:: GPU reset detected; invalidating the GPU state"<<std::endl ;
#endif
    mxDestroyArray(canary) ;
    canary = NULL ;
    Context::invalidateGpu() ;
  }
  return vl::vlSuccess ;
}

// Initialize GPU; also make sure that it was not reset by MATLAB
vl::Error
vl::MexContext::initGpu()
{
  validateGpu() ;
  if (!gpuIsInitialized) {
    mwSize dims = 1 ;
    mxInitGPU() ;
    // todo: can mxGPUCreateGPUArray return NULL ?
    mxGPUArray * gpuArray =
    mxGPUCreateGPUArray(1,&dims,mxINT8_CLASS,mxREAL,MX_GPU_DO_NOT_INITIALIZE) ;
    canary = mxGPUCreateMxArrayOnGPU(gpuArray) ;
    mexMakeArrayPersistent(canary) ;
    mxGPUDestroyGPUArray(gpuArray) ;
    gpuIsInitialized = true ;
  }
  return vl::vlSuccess ;
}
#endif

/* ---------------------------------------------------------------- */
/*                                                        MexTensor */
/* ---------------------------------------------------------------- */

/*
 The MexTensor class helps handling MATLAB CPU and GPU arrays.

 The design is somewhat ackward to match MATLAB assumpitons.

 The class can either:

 - wrap an existing mxArray (or mxArray + mxGPUArray)
 - or create a new mxArray (or mxArray + mxGPUArray)

 In the last case, the array is released when the destructor is
 called. However, this would normally interfere with MATLAB
 automatic garbage collection upon raising an exception (which
 can happen using mexErrMsgTxt() or, implicitly, when an array
 creation function cannot complete, for example due to a memory error).

 Therefore the constructors make the allocated memory persistent. C++
 guarantees that the arrays are freeed upon error in the destructors.

 Note that, upon cerating an array, errors such as running out of
 CPU/GPU memory can occurr. In this case, MATLAB throws an error
 and quits the MEX file (either implicitly or because we call
 mexErrMsgTxt()). Hence constructors always complete with a well
 defined object.

 */

/* ---------------------------------------------------------------- */
/* Constructing, clearing, destroying                               */
/* ---------------------------------------------------------------- */

vl::MexTensor::MexTensor(MexContext & context)
: context(context),
Tensor(),
array(NULL),
isArrayOwner(false)
#if ENABLE_GPU
, gpuArray(NULL)
#endif
{ }

mxArray *
vl::MexTensor::relinquish()
{
  if (isArrayOwner) {
    isArrayOwner = false ;
    return (mxArray*) array ;
  } else {
    // this is because we may be encapsulating an input argument
    // and we may be trying to return it
    // we should probably use the undocumented
    // extern mxArray *mxCreateSharedDataCopy(const mxArray *pr);
    return mxDuplicateArray(array) ;
  }
}

void
vl::MexTensor::clear()
{
#if ENABLE_GPU
  if (gpuArray) {
    mxGPUDestroyGPUArray(gpuArray) ;
    gpuArray = NULL ;
  }
#endif
  if (isArrayOwner) {
    if (array) {
      mxDestroyArray((mxArray*)array) ;
      array = NULL ;
    }
    isArrayOwner = false ;
  }
  memory = NULL ;
  memorySize = 0 ;
  deviceType = vl::CPU ;
  vl::TensorShape::clear() ;
}

vl::MexTensor::~MexTensor()
{
  clear() ;
}

/* ---------------------------------------------------------------- */
/* init with optional zero filling                                  */
/* ---------------------------------------------------------------- */


vl::Error
vl::MexTensor::initHelper(Device newDeviceType, Type newDataType,
                          TensorShape const & newShape, bool fillWithZeros)
{
  clear() ;

  // assign dimensions
  mwSize dimensions [VL_TENSOR_SHAPE_MAX_NUM_DIMENSIONS] ;
  for (int k = 0 ; k < newShape.getNumDimensions() ; ++k) {
    dimensions[k] = (mwSize)newShape.getDimension(k) ;
  }

  // compute the size in bytes
  mwSize newMemorySize = newShape.getNumElements() ;
  mxClassID classID ;
  switch (newDataType) {
    case vlTypeFloat:
      newMemorySize *= sizeof(DataTypeTraits<vlTypeFloat>::type) ;
      classID = mxSINGLE_CLASS ;
      break ;
#ifdef ENABLE_DOUBLE
    case vlTypeDouble:
      newMemorySize *= sizeof(DataTypeTraits<vlTypeDouble>::type) ;
      classID = mxDOUBLE_CLASS ;
      break ;
#endif
    default:
      abort() ;
  }

  // allocate the memory on CPU or GPU
  void * newMemory = NULL ;
  mxArray * newArray = NULL ;
#if ENABLE_GPU
  mxGPUArray* newGpuArray = NULL ;
#endif

  if (newDeviceType == vl::CPU) {
    if (fillWithZeros) {
      newArray = mxCreateNumericArray(4, dimensions, classID, mxREAL) ;
      newMemory = mxGetData(newArray) ;
    } else {
      mwSize dimensions_ [1] = {0} ;
      newMemory = mxMalloc(newMemorySize) ;
      newArray = mxCreateNumericArray(1, dimensions_,
                                      classID,
                                      mxREAL) ;
      mxSetData(newArray, newMemory) ;
      mxSetDimensions(newArray, dimensions, newShape.getNumDimensions()) ;
    }
  }
#ifdef ENABLE_GPU
  else {
    context.initGpu() ;
    newGpuArray = mxGPUCreateGPUArray(newShape.getNumDimensions(), dimensions,
                                      classID,
                                      mxREAL,
                                      fillWithZeros ? MX_GPU_INITIALIZE_VALUES : MX_GPU_DO_NOT_INITIALIZE) ;
    newArray = mxGPUCreateMxArrayOnGPU(newGpuArray) ;
    newMemory = mxGPUGetData(newGpuArray) ;
  }
#else
  else {
    abort() ;
  }
#endif

  //mexMakeArrayPersistent(newArray) ; // avoid double free with MATALB garbage collector upon error
  TensorShape::operator=(newShape) ;
  deviceType = newDeviceType ;
  dataType = newDataType ;
  memory = newMemory ;
  memorySize = newMemorySize ;
  array = newArray ;
  isArrayOwner = true ;
#if ENABLE_GPU
  gpuArray = newGpuArray ;
#endif
  return vl::vlSuccess ;
}

/* ---------------------------------------------------------------- */
/*                                                          init*() */
/* ---------------------------------------------------------------- */

vl::Error
vl::MexTensor::init(Device newDeviceType,
                    Type newDataType,
                    TensorShape const & newShape)
{
  return initHelper(newDeviceType, newDataType, newShape, false) ;
}

vl::Error
vl::MexTensor::initWithZeros(Device newDeviceType,
                             Type newDataType,
                             TensorShape const & newShape)
{
  return initHelper(newDeviceType, newDataType, newShape, true) ;
}

vl::Error
vl::MexTensor::initWithValue(Device newDeviceType,
                             Type newDataType,
                             TensorShape const & newShape,
                             double value)
{
  if (value == 0) {
    return initHelper(newDeviceType, newDataType, newShape, true) ;
  } else {
    vl::Error error = initHelper(newDeviceType, newDataType, newShape, false) ;
    if (error != vlSuccess) { return error ; }
    size_t const n = getNumElements() ;
    if (newDeviceType == vl::CPU) {
      switch (newDataType) {
        case vlTypeFloat: error = operations<vl::CPU,float>::fill((float*)memory, n, (float)value) ; break ;
#ifdef ENABLE_DOUBLE
        case vlTypeDouble: error = operations<vl::CPU,double>::fill((double*)memory, n, (double)value) ; break ;
#endif
        default: abort() ;
      }
    }
#ifdef ENABLE_GPU
    else {
      switch (newDataType) {
        case vlTypeFloat: error = operations<vl::GPU,float>::fill((float*)memory, n, (float)value) ; break ;
#ifdef ENABLE_DOUBLE
        case vlTypeDouble: error = operations<vl::GPU,double>::fill((double*)memory, n, (double)value) ; break ;
#endif
        default: abort() ;
      }
      if (error == vlErrorCuda) {
        cudaError_t error = cudaGetLastError() ;
        clear() ;
        mexErrMsgTxt((std::string("MexTensor: fill [CUDA error: ")
                      + cudaGetErrorString(error)
                      + "]"
                      ).c_str()) ;
      }
    }
#endif
  }
  return vl::vlSuccess ;
}

/* ---------------------------------------------------------------- */
/* init by wrapping a given array                                   */
/* ---------------------------------------------------------------- */

vl::Error
vl::MexTensor::init(mxArray const * array_)
{
  clear() ;
  if (array_ == NULL) { return vl::vlSuccess ; } // empty

  vl::Device newDeviceType ;
  vl::Type newDataType ;
  void const * newMemory = NULL ;
  mxArray * newArray = (mxArray*)array_ ;
#if ENABLE_GPU
  mxGPUArray* newGpuArray = NULL ;
#endif

  mwSize const * newDimensions ;
  mwSize newNumDimensions ;
  mxClassID newClassID ;

#ifdef ENABLE_GPU
  context.initGpu() ;
  if (mxIsGPUArray(array_)) {
    newDeviceType = GPU ;
    newGpuArray = (mxGPUArray*) mxGPUCreateFromMxArray(newArray) ;
    newMemory = mxGPUGetDataReadOnly(newGpuArray) ;
    newClassID = mxGPUGetClassID(newGpuArray) ;
    newDimensions = mxGPUGetDimensions(newGpuArray) ;
    newNumDimensions = mxGPUGetNumberOfDimensions(newGpuArray) ;
  } else
#endif
  {
    if (!mxIsNumeric(newArray)) {
      mexErrMsgTxt("An input is not a numeric array (or GPU support not compiled).") ;
    }
    newDeviceType = CPU ;
    newMemory = mxGetData(newArray) ;
    newClassID = mxGetClassID(newArray) ;
    newDimensions = mxGetDimensions(newArray) ;
    newNumDimensions = mxGetNumberOfDimensions(newArray) ;
  }

  if (newNumDimensions >= VL_TENSOR_SHAPE_MAX_NUM_DIMENSIONS) {
#if ENABLE_GPU
    if (newGpuArray) {
      mxGPUDestroyGPUArray(newGpuArray) ;
      newGpuArray = NULL ;
    }
#endif
    mexErrMsgTxt("An input has more than the maximum number of allowed dimensions.") ;
  }

  numDimensions = newNumDimensions ;
  for (int k = 0 ; k < numDimensions ; ++k) {
    setDimension(k, newDimensions[k]) ;
  }

  size_t newMemorySize = getNumElements() ;

  switch (newClassID) {
    case mxSINGLE_CLASS:
      newDataType = vlTypeFloat ;
      newMemorySize *= sizeof(DataTypeTraits<vlTypeFloat>::type) ;
      break ;

#ifdef ENABLE_DOUBLE
    case mxDOUBLE_CLASS:
      newDataType = vlTypeDouble ;
      newMemorySize *= sizeof(DataTypeTraits<vlTypeDouble>::type) ;
      break ;
#endif

    default:
      if (isEmpty()) {
        newDataType = vlTypeFloat ;
        newMemorySize = 0 ;
        break ;
      }
#ifdef ENABLE_DOUBLE
      mexErrMsgTxt("An input is neither SINGLE or DOUBLE nor it is empty.") ;
#else
      mexErrMsgTxt("An input is neither SINGLE nor empty.") ;
#endif
      break ;
  }

  deviceType = newDeviceType ;
  dataType = newDataType ;
  memory = (void*)newMemory ;
  memorySize = newMemorySize ;
  array = newArray ;
  isArrayOwner = false ;
#if ENABLE_GPU
  gpuArray = newGpuArray ;
#endif


  return vl::vlSuccess ;
}

size_t
vl::MexTensor::getMemorySize() const
{
  return memorySize ;
}

void vl::print(char const * str, vl::MexTensor const & tensor)
{
  size_t size = tensor.getMemorySize() ;
  double scaled ;
  size_t const * dimensions = tensor.getDimensions() ;
  const char * units ;
  const char * type ;
  if (size < 1024) {
    scaled = size ;
    units = "B" ;
  } else if (size < 1024*1024) {
    scaled = size / 1024.0 ;
    units = "KB" ;
  } else if (size < 1024*1024*1024) {
    scaled = size / (1024.0 * 1024.0) ;
    units = "MB" ;
  } else {
    scaled = size / (1024.0 * 1024.0 * 1024.0) ;
    units = "GB" ;
  }
  const char * dev = "" ;
  switch (tensor.getDeviceType()) {
    case vl::CPU : dev = "CPU" ; break ;
    case vl::GPU : dev = "GPU" ; break ;
  }
  switch (tensor.getDataType()) {
    case vl::vlTypeFloat: type = "float" ; break ;
    case vl::vlTypeDouble: type = "double" ; break ;
    case vl::vlTypeChar: type = "char" ; break ;
    default: type = "uknown type" ;
  }
  mexPrintf("%s[", str) ;
  for (int k = 0 ; k < tensor.getNumDimensions() ; ++k) {
    mexPrintf("%d ", dimensions[k]) ;
  }
  mexPrintf("| %s %.1f%s %s]\n",
            type,
            scaled,
            units,
            dev);
}
