/** @file    nnhelper.h
 ** @brief   MEX helper functions for CNNs.
 ** @author  Andrea Vedaldi
 **/

/*
Copyright (C) 2014 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_NNHELPER_H
#define VL_NNHELPER_H

#include "mex.h"
#ifdef ENABLE_GPU
#include "gpu/mxGPUArray.h"
#endif

#include <assert.h>

typedef struct PackedDataGeometry_
{
  mxClassID classID ;
  ptrdiff_t height ;
  ptrdiff_t width ;
  ptrdiff_t depth ;
  ptrdiff_t size ;
  ptrdiff_t numElements ;
} PackedDataGeometry ;

typedef enum PackedDataMode_
{
  empty,
  matlabArrayWrapper,
  matlabGpuArray,
  matlabMallocMemory,
  cudaMallocMemory
} PackedDataMode ;

typedef struct PackedData_
{
  PackedDataMode mode ;
  PackedDataGeometry geom ;
  mwSize memorySize ;
  float * memory ;
  mxArray * array ;
#ifdef ENABLE_GPU
  mxGPUArray * gpuArray ;
#endif
} PackedData ;

void
packed_data_geom_init (PackedDataGeometry * geom,
                       mxClassID classID,
                       size_t height,
                       size_t width,
                       size_t depth,
                       size_t size)
{
  geom->classID = classID ;
  geom->height = height ;
  geom->width = width ;
  geom->depth = depth ;
  geom->size = size ;
  geom->numElements = height*width*depth*size ;
}

void
packed_data_init_empty (PackedData * self)
{
  memset(self, 0, sizeof(PackedData)) ;
  self->mode = empty ;
  packed_data_geom_init(&self->geom,
                        mxSINGLE_CLASS,
                        0, 0, 0, 0) ;
}

void
packed_data_geom_display (PackedDataGeometry const * geom, char const * name)
{
  double const MB = 1024.0 * 1024.0 ;
  mexPrintf("%s: %d x %d x %d x %d [%.1f MB]\n",
            name,
            geom->height, geom->width, geom->depth, geom->size,
            geom->numElements * sizeof(float) / MB) ;
}

/*
 This function takes an array as input and initializes a corresponding PackedData structure.
 The structure will hold a pointer to the array. In GPU mode, the function expects the
 array to contain a GPU array; if so, a pointer to the latter is extracted as well.

 The self->isOwner flag is set to @c false to indicate the fact that the structure
 is just a wrapper around an existing MATLAB array.
 */

void
packed_data_init_with_array (PackedData * map, bool gpuMode, mxArray const* array)
{
  mwSize const * dimensions ;
  mwSize numDimensions ;
  mxClassID classID ;
  packed_data_init_empty(map) ;

#ifdef ENABLE_GPU
  if (gpuMode) {
    if (!mxIsGPUArray(array)) {
      mexErrMsgTxt("The inputs are of mixed GPU and CPU types.") ;
    }
    map->mode = matlabGpuArray ;
    map->gpuArray = (mxGPUArray*) mxGPUCreateFromMxArray(array) ;
    map->memory = (float*) mxGPUGetDataReadOnly(map->gpuArray) ;
    classID = mxGPUGetClassID(map->gpuArray) ;
    dimensions = mxGPUGetDimensions(map->gpuArray) ;
    numDimensions = mxGPUGetNumberOfDimensions(map->gpuArray) ;
  } else
#endif
  {
    if (!mxIsNumeric(array)) {
      mexErrMsgTxt("The inputs are neither all numeric CPU arrays or GPU arrays.") ;
    }
    map->mode = matlabArrayWrapper ;
    map->array = (mxArray*) array ;
    map->memory = (float*) mxGetData(map->array) ;
    classID = mxGetClassID(map->array) ;
    dimensions = mxGetDimensions(map->array) ;
    numDimensions = mxGetNumberOfDimensions(map->array) ;
  }
  packed_data_geom_init(&map->geom,
                        classID,
                        (numDimensions >= 1) ? dimensions[0] : 1,
                        (numDimensions >= 2) ? dimensions[1] : 1,
                        (numDimensions >= 3) ? dimensions[2] : 1,
                        (numDimensions >= 4) ? dimensions[3] : 1) ;
  map->memorySize = map->geom.numElements * sizeof(float) ;
}

/*
 This function initializes a PackedData structure from a desired data geometry:

 - In CPU mode, the function allocates a MATLAB array (self->array).
 - In GPU mode, the function allocates a MATLAB GPU array (self->gpuArray).

 The flag self->isOwner is set to @c true to indicate that the data was
 allocated here. If @c initialize is @c true, then the data is zeroed.
 */

void
packed_data_init_with_geom (PackedData * map,
                            bool gpuMode,
                            PackedDataGeometry geom,
                            bool persistent,
                            bool initialize,
                            float value)
{
  assert(geom.classID == mxSINGLE_CLASS) ;
  float * memory = NULL ;

  packed_data_init_empty(map) ;
  map->geom = geom ;
  map->memorySize = map->geom.numElements * sizeof(float) ;

  /* create a CPU array with the specified values */
  if (! (gpuMode && (!initialize || value == 0))) {
    if (!initialize) {
      memory = (float*)mxMalloc(map->memorySize) ;
    } else {
      if (value == 0) {
        memory = (float*)mxCalloc(1, map->memorySize) ;
      } else {
        memory = (float*)mxMalloc(map->memorySize) ;
        for (int i = 0 ; i < geom.numElements ; ++i) { memory[i] = value ; }
      }
    }
  }

#ifdef ENABLE_GPU
  if (gpuMode) {
    cudaError_t err ;
    if (!persistent) {
      /* if not persistent, create a GPU array */
      mwSize dimensions [4] = {geom.height, geom.width, geom.depth, geom.size} ;
      map->gpuArray = mxGPUCreateGPUArray
      (4, dimensions, mxSINGLE_CLASS, mxREAL,
       (initialize && value == 0) ? MX_GPU_INITIALIZE_VALUES : MX_GPU_DO_NOT_INITIALIZE) ;
      map->mode = matlabGpuArray ;
      map->memory = (float*) mxGPUGetData(map->gpuArray) ;
      if (initialize && value != 0) {
        err = cudaMemcpy(map->memory, memory, map->memorySize, cudaMemcpyHostToDevice) ;
        if (err != cudaSuccess) {
          mexPrintf("cudaMemcpy: error (%s)\n", cudaGetErrorString(err)) ;
        }
      }
    } else {
      /* if persistent, use CUDA to allocate the memory (MATLAB does not have persistent GPU arrays) */
      map->mode = cudaMallocMemory ;
      err = cudaMalloc((void**)&map->memory, map->memorySize) ;
      if (err != cudaSuccess) {
        mexPrintf("cudaMemcpy: error (%s)\n", cudaGetErrorString(err)) ;
      }
      if (initialize) {
        if (value == 0) {
          err = cudaMemset(map->memory, 0, map->memorySize) ;
          if (err != cudaSuccess) {
            mexPrintf("cudaMemset: error (%s)\n", cudaGetErrorString(err)) ;
          }
        } else {
          err = cudaMemcpy(map->memory, memory, map->memorySize, cudaMemcpyHostToDevice) ;
          if (err != cudaSuccess) {
            mexPrintf("cudaMemcpy: error (%s)\n", cudaGetErrorString(err)) ;
          }
        }
      }
    }
    if (memory) { mxFree(memory) ; memory = NULL ; }
  } else
#endif
  {
    map->mode = matlabMallocMemory ;
    map->memory = memory ;
    if (persistent) { mexMakeMemoryPersistent(map->memory) ; }
  }
}

/*
 This function deinits a packed data structure. It does the following:
 */

void packed_data_deinit (PackedData * map)
{
  switch (map->mode) {
    case empty: break ;
    case matlabArrayWrapper : break ;
    case matlabMallocMemory : mxFree(map->memory) ; break ;
#ifdef ENABLE_GPU
    case cudaMallocMemory : cudaFree(map->memory) ; break ;
    case matlabGpuArray : mxGPUDestroyGPUArray(map->gpuArray) ; break ;
#endif
    default: assert(false) ;
  }
  packed_data_init_empty(map) ;
}

/*
 This function deinits a packed data structure returing the contained
 array.

 It does the following:

 - If the data contains a self->gpuArray, it first check whetehr the
 data contains a self->array too. In the latter case, then
 it simply destroys the gpuArray under the assumption that
 self->array already holds a copy to it. Otherwise, it creates
 an array coping the gpuArray, and only then destrops the latter.

 - Otherwise, it simply returns the array.

 - In all cases, it reset the data structures.
*/

mxArray* packed_data_deinit_extracting_array(PackedData * map)
{
  mxArray* array ;
  switch (map->mode) {
    case empty : assert(false) ; break ;
    case matlabArrayWrapper : assert(false) ; break ;
    case cudaMallocMemory : assert(false) ; break ;
    case matlabMallocMemory :
    {
      mwSize dimensions [4] = {map->geom.height, map->geom.width, map->geom.depth, map->geom.size} ;
      mwSize dimensions_ [4] = {0,0,0,0} ;
      array = mxCreateNumericArray(4, dimensions_, mxSINGLE_CLASS, mxREAL) ;
      mxSetData(array, map->memory) ;
      mxSetDimensions(array, dimensions, 4) ;
      map->mode = empty ;
      map->memory = NULL ;
      break ;
    }
    case matlabGpuArray :
    {
#ifdef ENABLE_GPU
      array = mxGPUCreateMxArrayOnGPU(map->gpuArray) ;
#endif
      break ;
    }
  }
  packed_data_deinit(map) ;
  return array;
}

#endif
