// @file datamex.cu
// @brief Basic data structures (MEX support)
// @author Andrea Vedaldi

/*
Copyright (C) 2015 Andrea Vedaldi.
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

using namespace vl ;

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
  // so that ~Context does not crash if MATLAB reset the GPU in the mean time
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
  memoryType = vl::CPU ;
  width = 0 ;
  height = 0 ;
  depth = 0 ;
  size = 0 ;
}

vl::MexTensor::~MexTensor()
{
  clear() ;
}

/* ---------------------------------------------------------------- */
/* init without filling                                             */
/* ---------------------------------------------------------------- */

vl::Error
vl::MexTensor::init(Device dev, TensorGeometry const & geom)
{
  mwSize dimensions [4] = {(mwSize)geom.getHeight(),
                           (mwSize)geom.getWidth(),
                           (mwSize)geom.getDepth(),
                           (mwSize)geom.getSize()} ;
  mwSize newMemorySize = geom.getNumElements() * sizeof(float) ;
  float * newMemory = NULL ;
  mxArray * newArray = NULL ;
#if ENABLE_GPU
  mxGPUArray* newGpuArray = NULL ;
#endif

  if (dev == vl::CPU) {
    mwSize dimensions_ [4] = {0} ;
    newMemory = (float*)mxMalloc(newMemorySize) ;
    newArray = mxCreateNumericArray(4, dimensions_, mxSINGLE_CLASS, mxREAL) ;
    mxSetData(newArray, newMemory) ;
    mxSetDimensions(newArray, dimensions, 4) ;
  }

#ifdef ENABLE_GPU
  else {
    newGpuArray = mxGPUCreateGPUArray(4, dimensions,
                                      mxSINGLE_CLASS, mxREAL,
                                      MX_GPU_DO_NOT_INITIALIZE) ;
    newArray = mxGPUCreateMxArrayOnGPU(newGpuArray) ;
    newMemory = (float*) mxGPUGetData(newGpuArray) ;
  }
#else
  else {
    abort() ;
  }
#endif

  //mexMakeArrayPersistent(newArray) ; // avoid double free with MATALB garbage collector upon error
  TensorGeometry::operator=(geom) ;
  memoryType = dev ;
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
/* init filling with zeros                                          */
/* ---------------------------------------------------------------- */

vl::Error
vl::MexTensor::initWithZeros(vl::Device dev, TensorGeometry const & geom)
{

  clear() ;

  mwSize dimensions [4] = {(mwSize)geom.getHeight(),
                           (mwSize)geom.getWidth(),
                           (mwSize)geom.getDepth(),
                           (mwSize)geom.getSize()} ;
  mwSize newMemorySize = geom.getNumElements() * sizeof(float) ;
  float * newMemory = NULL ;
  mxArray * newArray = NULL ;
#if ENABLE_GPU
  mxGPUArray* newGpuArray = NULL ;
#endif

  if (dev == vl::CPU) {
    newArray = mxCreateNumericArray(4, dimensions, mxSINGLE_CLASS, mxREAL) ;
    newMemory = (float*) mxGetData(newArray) ;
  }

#ifdef ENABLE_GPU
  else {
    context.initGpu() ;
    newGpuArray = mxGPUCreateGPUArray(4, dimensions, mxSINGLE_CLASS,
                                      mxREAL, MX_GPU_INITIALIZE_VALUES) ;
    newArray = mxGPUCreateMxArrayOnGPU(newGpuArray) ;
    newMemory = (float*) mxGPUGetData((mxGPUArray*)newGpuArray) ;
  }
#else
  else {
    abort() ;
  }
#endif

  //mexMakeArrayPersistent(newArray) ; // avoid double free with MATALB garbage collector upon error
  TensorGeometry::operator=(geom) ;
  memoryType = dev ;
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
/* init with any fill                                               */
/* ---------------------------------------------------------------- */

#if ENABLE_GPU
template<typename type> __global__ void
fill (type * data, type value, int size)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x ;
  if (index < size) data[index] = value ;
}
#endif

vl::Error
vl::MexTensor::init(vl::Device dev, vl::TensorGeometry const & geom, float value)
{
  if (value == 0) {
    initWithZeros(dev, geom) ;
  } else {
    init(dev, geom) ;
    if (memoryType == vl::CPU) {
      int const n = getNumElements() ;
      for (int i = 0 ; i < n ; ++i) { memory[i] = value ; }
    }
#ifdef ENABLE_GPU
    else {
      fill<float>
      <<<divideUpwards(getNumElements(), VL_CUDA_NUM_THREADS),
        VL_CUDA_NUM_THREADS>>>
        ((float*)getMemory(), value, getNumElements()) ;
      cudaError_t error = cudaGetLastError() ;
      if (error != cudaSuccess) {
        clear() ;
        mexErrMsgTxt((std::string("MexTensor: fill: CUDA error: ")
                      + cudaGetErrorString(error)).c_str()) ;
      }
    }
#endif
  }
  return vl::vlSuccess ;
}

/* ---------------------------------------------------------------- */
/* init with array                                                  */
/* ---------------------------------------------------------------- */

vl::Error
vl::MexTensor::init(mxArray const * array_)
{
  clear() ;
  if (array_ == NULL) { return vl::vlSuccess ; } // empty

  vl::Device dev ;
  float * newMemory = NULL ;
  mxArray * newArray = (mxArray*)array_ ;
#if ENABLE_GPU
  mxGPUArray* newGpuArray = NULL ;
#endif

  mwSize const * dimensions ;
  mwSize numDimensions ;
  mxClassID classID ;

#ifdef ENABLE_GPU
  context.initGpu() ;
  if (mxIsGPUArray(array_)) {
    dev = GPU ;
    newGpuArray = (mxGPUArray*) mxGPUCreateFromMxArray(newArray) ;
    newMemory = (float*) mxGPUGetDataReadOnly(newGpuArray) ;
    classID = mxGPUGetClassID(newGpuArray) ;
    dimensions = mxGPUGetDimensions(newGpuArray) ;
    numDimensions = mxGPUGetNumberOfDimensions(newGpuArray) ;
  } else
#endif

  {
    if (!mxIsNumeric(newArray)) {
      mexErrMsgTxt("An input is not a numeric array (or GPU support not compiled).") ;
    }
    dev = CPU ;
    newMemory = (float*) mxGetData(newArray) ;
    classID = mxGetClassID(newArray) ;
    dimensions = mxGetDimensions(newArray) ;
    numDimensions = mxGetNumberOfDimensions(newArray) ;
  }

  height = (numDimensions >= 1) ? dimensions[0] : 1 ;
  width  = (numDimensions >= 2) ? dimensions[1] : 1 ;
  depth  = (numDimensions >= 3) ? dimensions[2] : 1 ;
  size   = (numDimensions >= 4) ? dimensions[3] : 1 ;
  memoryType = dev ;
  memory = newMemory ;
  memorySize = getNumElements() * sizeof(float) ;
  array = newArray ;
  isArrayOwner = false ;
#if ENABLE_GPU
  gpuArray = newGpuArray ;
#endif

  if (classID != mxSINGLE_CLASS && ! isEmpty()) {
    mexErrMsgTxt("An input is not a SINGLE array nor it is empty.") ;
  }
  return vl::vlSuccess ;
}

void vl::print(char const * str, vl::Tensor const & tensor)
{
  size_t size = tensor.getNumElements() * sizeof(float) ;
  double scaled ;
  const char * units ;
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
  switch (tensor.getMemoryType()) {
    case vl::CPU : dev = "CPU" ; break ;
    case vl::GPU : dev = "GPU" ; break ;
  }
  mexPrintf("%s[%d x %d x %d x %d | %.1f%s %s]\n",
            str,
            tensor.getHeight(),
            tensor.getWidth(),
            tensor.getDepth(),
            tensor.getSize(),
            scaled,
            units,
            dev);
}
