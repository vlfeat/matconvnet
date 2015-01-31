//
//  datamex.cpp
//  matconv
//
//  Created by Andrea Vedaldi on 31/01/2015.
//  Copyright (c) 2015 Andrea Vedaldi. All rights reserved.
//

#include "datamex.hpp"

using namespace vl ;


void vl::print(char const * str, vl::TensorGeometry const & tensor)
{
  mexPrintf("%s[%d x %d x %d x %d] (%.2f MB)\n",
            str,
            tensor.getHeight(),
            tensor.getWidth(),
            tensor.getDepth(),
            tensor.getSize(),
            tensor.getNumElements() * sizeof(float) / (1024.0 * 1024.0)) ;
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
      float * buffer = (float*)mxMalloc(memorySize) ;
      for (int i = 0 ; i < getNumElements() ; ++i) { memory[i] = value ; }
      cudaError_t err = cudaMemcpy(memory, buffer, memorySize, cudaMemcpyHostToDevice) ;
      if (err != cudaSuccess) {
        mexPrintf("cudaMemcpy: error (%s)\n", cudaGetErrorString(err)) ;
      }
      mxFree(buffer) ;
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
  if (memoryType == vl::CPU) {
    mwSize dimensions_ [4] = {0} ;
    memory = (float*)mxMalloc(memorySize) ;
    array = mxCreateNumericArray(4, dimensions_, mxSINGLE_CLASS, mxREAL) ;
    mxSetData((mxArray*)array, memory) ;
    mxSetDimensions((mxArray*)array, dimensions, 4) ;
  }
#ifdef ENABLE_GPU
  else {
    gpuArray = mxGPUCreateGPUArray(4, dimensions, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE) ;
    array = mxGPUCreateMxArrayOnGPU(gpuArray) ;
    memory = (float*) mxGPUGetData((mxGPUArray*)gpuArray) ;
  }
#endif
  isArrayOwner = true ;
}

/* ---------------------------------------------------------------- */
/* allocInitialized                                                 */
/* ---------------------------------------------------------------- */

void vl::MexTensor::allocInitialized()
{
  mwSize dimensions [4] = {height, width, depth, size} ;
  memorySize = getNumElements() * sizeof(float) ;
  if (memoryType == vl::CPU) {
    array = mxCreateNumericArray(4, dimensions, mxSINGLE_CLASS, mxREAL) ;
    memory = (float*) mxGetData(array) ;
  }
#ifdef ENABLE_GPU
  else {
    gpuArray = mxGPUCreateGPUArray(4, dimensions, mxSINGLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES) ;
    array = mxGPUCreateMxArrayOnGPU(gpuArray) ;
    memory = (float*) mxGPUGetData((mxGPUArray*)gpuArray) ;
  }
#endif
  isArrayOwner = true ;
}





