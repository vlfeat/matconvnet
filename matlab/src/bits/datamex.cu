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

using namespace vl ;

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


/* transfer ownership (a bit like auto_ptr) */
MexTensor &
vl::MexTensor::operator= (MexTensor & tensor)
{
  if (&tensor == this) return *this;
  clear() ;
  width = tensor.width ;
  height = tensor.height ;
  depth = tensor.depth ;
  size = tensor.size ;
  memory = tensor.memory ;
  memoryType = tensor.memoryType;
  memorySize = tensor.memorySize ;
  array = tensor.array ;
  isArrayOwner = tensor.isArrayOwner ;
#if ENABLE_GPU
  gpuArray = tensor.gpuArray ;
#endif
  tensor.width = 0 ;
  tensor.height = 0 ;
  tensor.depth = 0 ;
  tensor.size = 0 ;
  tensor.memory = 0 ;
  tensor.memorySize = 0 ;
  tensor.array = NULL ;
  tensor.isArrayOwner = false ;
#if ENABLE_GPU
  tensor.gpuArray = NULL ;
#endif
  return *this ;
}

/* ---------------------------------------------------------------- */
/* MexTensor()                                                      */
/* ---------------------------------------------------------------- */

vl::MexTensor::MexTensor()
: Tensor(), array(NULL), isArrayOwner(false)
#if ENABLE_GPU
, gpuArray(NULL)
#endif
{ }

/* ---------------------------------------------------------------- */
/* MexTensor(array)                                                 */
/* ---------------------------------------------------------------- */

vl::MexTensor::MexTensor(mxArray const * array_)
: array(array_), isArrayOwner(false)
#if ENABLE_GPU
, gpuArray(NULL)
#endif
{
  mwSize const * dimensions ;
  mwSize numDimensions ;
  mxClassID classID ;

  if (array_ == NULL) { return ; } // empty

#ifdef ENABLE_GPU
  if (mxIsGPUArray(array)) {
    memoryType = GPU ;
    gpuArray = (mxGPUArray*) mxGPUCreateFromMxArray(array) ;
    memory = (float*) mxGPUGetDataReadOnly(gpuArray) ;
    classID = mxGPUGetClassID(gpuArray) ;
    dimensions = mxGPUGetDimensions(gpuArray) ;
    numDimensions = mxGPUGetNumberOfDimensions(gpuArray) ;
  } else
#endif
  {
    if (!mxIsNumeric(array)) {
      mexErrMsgTxt("An input is not a numeric array (or GPU support not compiled).") ;
    }
    memoryType = CPU ;
    memory = (float*) mxGetData(array) ;
    classID = mxGetClassID(array) ;
    dimensions = mxGetDimensions(array) ;
    numDimensions = mxGetNumberOfDimensions(array) ;
  }

  height = (numDimensions >= 1) ? dimensions[0] : 1 ;
  width  = (numDimensions >= 2) ? dimensions[1] : 1 ;
  depth  = (numDimensions >= 3) ? dimensions[2] : 1 ;
  size   = (numDimensions >= 4) ? dimensions[3] : 1 ;

  memorySize = getNumElements() * sizeof(float) ;

  if (classID != mxSINGLE_CLASS && !isEmpty()) {
    mexErrMsgTxt("An input is not a SINGLE array nor it is empty.") ;
  }
}

/* ---------------------------------------------------------------- */
/* MexTensor(nofill)                                                */
/* ---------------------------------------------------------------- */

vl::MexTensor::MexTensor(vl::Device type, vl::TensorGeometry const & geom)
: Tensor(NULL, 0, type, geom),
array(NULL),
isArrayOwner(false)
#if ENABLE_GPU
, gpuArray(NULL)
#endif
{
  allocUninitialized() ;
}

/* ---------------------------------------------------------------- */
/* MexTensor(fill)                                                  */
/* ---------------------------------------------------------------- */


#if ENABLE_GPU
template<typename type> __global__ void
fill (type * data, type value, int size)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x ;
  if (index < size) data[index] = value ;
}
#endif

vl::MexTensor::MexTensor(vl::Device type, vl::TensorGeometry const & geom, float value)
: Tensor(NULL, 0, type, geom),
array(NULL),
isArrayOwner(false)
#if ENABLE_GPU
, gpuArray(NULL)
#endif
{
  if (value == 0) {
    allocInitialized() ;
  } else {
    allocUninitialized() ;
    if (memoryType == vl::CPU) {
      for (int i = 0 ; i < getNumElements() ; ++i) { memory[i] = value ; }
    }
#ifdef ENABLE_GPU
    else {
      fill<float>
      <<<divideUpwards(getNumElements(), VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS>>>
      ((float*)memory, getNumElements(), value) ;
      cudaError_t error = cudaGetLastError() ;
      if (error != cudaSuccess) {
        clear() ;
        mexErrMsgTxt((std::string("MexTensor: fill: CUDA error: ") + cudaGetErrorString(error)).c_str()) ;
      }
    }
#endif
  }
}

/* ---------------------------------------------------------------- */
/* relinquish()                                                     */
/* ---------------------------------------------------------------- */

mxArray * vl::MexTensor::relinquish()
{
  isArrayOwner = false ;
  return (mxArray*) array ;
}

/* ---------------------------------------------------------------- */
/* clear()                                                          */
/* ---------------------------------------------------------------- */

void vl::MexTensor::clear()
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
  width = 0 ;
  height = 0 ;
  depth = 0 ;
  size = 0 ;
}

/* ---------------------------------------------------------------- */
/* ~MexTensor()                                                     */
/* ---------------------------------------------------------------- */

vl::MexTensor::~MexTensor()
{
  clear() ;
}

/* ---------------------------------------------------------------- */
/* allocUninitialized                                               */
/* ---------------------------------------------------------------- */

void vl::MexTensor::allocUninitialized()
{
  mwSize dimensions [4] = {height, width, depth, size} ;
  memorySize = getNumElements() * sizeof(float) ;
  mxArray * newArray ;
  if (memoryType == vl::CPU) {
    mwSize dimensions_ [4] = {0} ;
    memory = (float*)mxMalloc(memorySize) ;
    newArray = mxCreateNumericArray(4, dimensions_, mxSINGLE_CLASS, mxREAL) ;
    mxSetData(newArray, memory) ;
    mxSetDimensions(newArray, dimensions, 4) ;
  }
#ifdef ENABLE_GPU
  else {
    gpuArray = mxGPUCreateGPUArray(4, dimensions, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE) ;
    newArray = mxGPUCreateMxArrayOnGPU(gpuArray) ;
    memory = (float*) mxGPUGetData((mxGPUArray*)gpuArray) ;
  }
#endif
  //mexMakeArrayPersistent(newArray) ; // avoid double free with MATALB garbage collector upon error
  array = newArray ;
  isArrayOwner = true ;
}

/* ---------------------------------------------------------------- */
/* allocInitialized                                                 */
/* ---------------------------------------------------------------- */

void vl::MexTensor::allocInitialized()
{
  mwSize dimensions [4] = {height, width, depth, size} ;
  memorySize = getNumElements() * sizeof(float) ;
  mxArray * newArray ;
  if (memoryType == vl::CPU) {
    newArray = mxCreateNumericArray(4, dimensions, mxSINGLE_CLASS, mxREAL) ;
    memory = (float*) mxGetData(newArray) ;
  }
#ifdef ENABLE_GPU
  else {
    gpuArray = mxGPUCreateGPUArray(4, dimensions, mxSINGLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES) ;
    newArray = mxGPUCreateMxArrayOnGPU(gpuArray) ;
    memory = (float*) mxGPUGetData((mxGPUArray*)gpuArray) ;
  }
#endif
  //mexMakeArrayPersistent(newArray) ; // avoid double free with MATALB garbage collector upon error
  array = newArray ;
  isArrayOwner = true ;
}




