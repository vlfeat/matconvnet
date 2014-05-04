/** @file gconv.cu
 ** @brief Convolution block
 ** @author Andrea Vedaldi
 **/

#include "mex.h"
#ifdef ENABLE_GPU
#include "gpu/mxGPUArray.h"
#endif
#include "bits/mexutils.h"
#include "bits/im2col.hpp"

#include <blas.h>
#include <iostream>
#include <assert.h>

#ifdef ENABLE_GPU
#include <cublas_v2.h>
#endif

/* option codes */
enum {
  opt_stride = 0,
  opt_pad,
  opt_verbose
} ;

/* options */
vlmxOption  options [] = {
  {"Stride",           1,   opt_stride            },
  {"Pad",              1,   opt_pad               },
  {"Verbose",          0,   opt_verbose           },
  {0,                  0,   0                     }
} ;

/* ---------------------------------------------------------------- */
/*                                                 Helper functions */
/* ---------------------------------------------------------------- */

typedef struct FeatureMapGeometry_
{
  mxClassID classID ;
  ptrdiff_t height ;
  ptrdiff_t width ;
  ptrdiff_t depth ;
  ptrdiff_t size ;
  ptrdiff_t numElements ;
} FeatureMapGeometry ;

typedef struct FeatureMap_
{
  bool isOwner ;
  mxArray * array ;
#ifdef ENABLE_GPU
  mxGPUArray * gpuArray ;
#endif
  FeatureMapGeometry geom ;
} FeatureMap ;

void feature_map_display (FeatureMap const * map, char const * name)
{
  double const MB = 1024.0 * 1024.0 ;
  mexPrintf("vl_nnconv: %s: %d x %d x %d x %d [%.1f MB]\n",
            name,
            map->geom.height, map->geom.width, map->geom.depth, map->geom.size,
            (double)(map->geom.numElements*4)/MB) ;
}

void feature_map_init_with_array (FeatureMap * map, bool gpuMode, mxArray const* array)
{
  mwSize const * dimensions ;
  mwSize numDimensions ;

#ifndef ENABLE_GPU
  assert(!gpuMode) ;
#endif

  map->isOwner = false ;
  map->array = (mxArray*)array ;
#ifdef ENABLE_GPU
  map->gpuArray = NULL ;
  if (gpuMode) {
    if (!mxIsGPUArray(map->array)) {
      mexErrMsgTxt("The inputs are of mixed GPU and CPU types.") ;
    }
    map->gpuArray = (mxGPUArray*) mxGPUCreateFromMxArray(map->array) ;
    map->geom.classID = mxGPUGetClassID(map->gpuArray) ;
    map->geom.numElements = mxGPUGetNumberOfElements(map->gpuArray) ;
    dimensions = mxGPUGetDimensions(map->gpuArray) ;
    numDimensions = mxGPUGetNumberOfDimensions(map->gpuArray) ;
  } else
#endif
  {
    if (!mxIsNumeric(map->array)) {
      mexErrMsgTxt("The inputs are neither all numeric CPU arrays or GPU arrays.") ;
    }
    map->geom.classID = mxGetClassID(map->array) ;
    map->geom.numElements = mxGetNumberOfElements(map->array) ;
    dimensions = mxGetDimensions(map->array) ;
    numDimensions = mxGetNumberOfDimensions(map->array) ;
  }
  map->geom.height = (numDimensions >= 1) ? dimensions[0] : 1 ;
  map->geom.width = (numDimensions >= 2) ? dimensions[1] : 1 ;
  map->geom.depth = (numDimensions >= 3) ? dimensions[2] : 1 ;
  map->geom.size = (numDimensions >= 4) ? dimensions[3] : 1 ;
}

void feature_map_init_with_geom (FeatureMap * map, bool gpuMode,
                                 FeatureMapGeometry geom, bool initialize)
{
  mwSize dimensions [4] = {geom.height, geom.width, geom.depth, geom.size} ;
  map->isOwner = true ;
  map->geom = geom ;
  map->array = NULL ;
#ifdef ENABLE_GPU
  map->gpuArray = NULL ;
  if (gpuMode) {
    map->gpuArray = mxGPUCreateGPUArray(4, dimensions, mxSINGLE_CLASS, mxREAL,
                                        (initialize)?MX_GPU_INITIALIZE_VALUES:MX_GPU_DO_NOT_INITIALIZE) ;
  } else
#endif
  {
    map->array = mxCreateNumericArray(4, dimensions, mxSINGLE_CLASS, mxREAL) ;
  }
}

void feature_map_init_with_geom_and_ones (FeatureMap * map, bool gpuMode,
                                          FeatureMapGeometry geom)
{
  mwSize dimensions [4] = {geom.height, geom.width, geom.depth, geom.size} ;
  map->isOwner = true ;
  map->geom = geom ;
  map->array = mxCreateNumericArray(4, dimensions, mxSINGLE_CLASS, mxREAL) ;
  int i ;
  float* data = (float*)mxGetData(map->array) ;
  for (i = 0 ; i < geom.numElements ; ++i) { data[i] = 1.0f ; }
#ifdef ENABLE_GPU
  map->gpuArray = NULL ;
  if (gpuMode) {
    map->gpuArray = (mxGPUArray*) mxGPUCreateFromMxArray (map->array) ;
  }
#endif
}

void feature_map_deinit (FeatureMap * map)
{
#ifdef ENABLE_GPU
  if (map->gpuArray) {
    mxGPUDestroyGPUArray(map->gpuArray) ;
    map->gpuArray = NULL ;
  }
#endif
  if (map->isOwner && map->array) {
    mxDestroyArray(map->array) ;
  }
  map->array = NULL ;
}

mxArray* feature_map_deinit_extracting_array(FeatureMap * map)
{
  mxArray* array = map->array ;
  map->array = NULL ;
#ifdef ENABLE_GPU
  if (map->gpuArray) {
    if (!array) {
      array = mxGPUCreateMxArrayOnGPU(map->gpuArray) ;
    }
    mxGPUDestroyGPUArray(map->gpuArray) ;
    map->gpuArray = NULL ;
  }
#endif
  return array ;
}

/* ---------------------------------------------------------------- */
/*                                                       MEX driver */
/* ---------------------------------------------------------------- */

enum {
  IN_DATA = 0, IN_FILTERS, IN_BIASES, IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_DERFILTERS, OUT_DERBIASES, OUT_END
} ;

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  /* inputs */
  FeatureMap data ;
  FeatureMap filters ;
  FeatureMap biases ;
  FeatureMap derOutput ;

  /* temporary buffers */
  FeatureMap temp ;
  FeatureMap allOnes ;

  /* outputs */
  FeatureMap output ;
  FeatureMap derData ;
  FeatureMap derFilters ;
  FeatureMap derBiases ;

  int stride = 1 ;
  int pad = 0 ;
  int numGroups = 1 ;

#if ENABLE_GPU
  cublasStatus_t stat;
  cublasHandle_t handle;
  bool gpuMode = false ;
#else
  bool const gpuMode = false ;
#endif
  bool backMode = false ;
  bool biasMode = false ;
  bool fullyConnectedMode = false ;
  bool computeOutput = true ;
  bool computeDerData = true ;
  bool computeDerFilters = true ;
  bool computeDerBiases = true ;

  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  if (nin < 3) {
    mexErrMsgTxt("There are less than three arguments.") ;
  }

  if (nin > 3 && vlmxIsString(in[3],-1)) {
    next = 3 ;
    backMode = 0 ;
  } else {
    backMode = (nin >= 4) ;
  }

  biasMode = (mxGetNumberOfElements(in[IN_BIASES]) > 0) ;

#if ENABLE_GPU
  gpuMode = mxIsGPUArray(in[IN_DATA]) ;
  if (gpuMode) {
    mxInitGPU() ;
    stat = cublasCreate(&handle) ;
    if (stat != CUBLAS_STATUS_SUCCESS) {
      mexErrMsgTxt("Could not initialize cuBLAS.") ;
    }
  }
#else
  if (!mxIsNumeric(in[IN_DATA])) {
    mexErrMsgTxt("DATA must be numeric (note: GPU support not compiled).") ;
  }
#endif

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
      case opt_verbose :
        ++ verbosity ;
        break ;

      case opt_stride :
        if (!vlmxIsPlainScalar(optarg) || (stride = (int) *mxGetPr(optarg)) < 1) {
          mexErrMsgTxt("STRIDE must be a positive integer.") ;
        }
        break ;

      case opt_pad :
        if (!vlmxIsPlainScalar(optarg) || (pad = (int) *mxGetPr(optarg)) < 0) {
          mexErrMsgTxt("PAD must be a non-negative integer.") ;
        }
        break ;

      default: break ;
    }
  }

  feature_map_init_with_array (&data, gpuMode, in[IN_DATA]) ;
  feature_map_init_with_array (&filters, gpuMode, in[IN_FILTERS]) ;
  if (biasMode) { feature_map_init_with_array(&biases, gpuMode, in[IN_BIASES]) ; }
  if (backMode) { feature_map_init_with_array(&derOutput, gpuMode, in[IN_DEROUTPUT]) ; }

  if (data.geom.classID != mxSINGLE_CLASS) {
    mexErrMsgTxt("DATA is not of class SINGLE.");
  }
  if (filters.geom.classID != mxSINGLE_CLASS) {
    mexErrMsgTxt("FILTERS is not of class SINGLE.");
  }
  if (biasMode && (biases.geom.classID != mxSINGLE_CLASS)) {
    mexErrMsgTxt("BIASES is not of class SINGLE.");
  }
  if (backMode && (derOutput.geom.classID != mxSINGLE_CLASS)) {
    mexErrMsgTxt("DEROUTPUT is not of class SINGLE.");
  }

  output.geom.height = (data.geom.height + 2*pad - filters.geom.height)/stride + 1 ;
  output.geom.width = (data.geom.width + 2*pad - filters.geom.width)/stride + 1 ;
  output.geom.depth = filters.geom.size ;
  output.geom.size = data.geom.size ;
  output.geom.numElements = output.geom.height*output.geom.width*output.geom.depth*output.geom.size ;
  derData.geom = data.geom ;
  derFilters.geom = filters.geom ;
  if (biasMode) {
    allOnes.geom.height = output.geom.height ;
    allOnes.geom.width = output.geom.width ;
    allOnes.geom.depth = 1 ;
    allOnes.geom.size = 1 ;
    allOnes.geom.numElements = allOnes.geom.height*allOnes.geom.width*allOnes.geom.depth*allOnes.geom.size ;
    derBiases.geom = biases.geom ;
  }

  /* grouped filters */
  numGroups = data.geom.depth / filters.geom.depth ;

  temp.geom.height = output.geom.height ;
  temp.geom.width = output.geom.width ;
  temp.geom.depth = filters.geom.height*filters.geom.width*filters.geom.depth*numGroups ;
  temp.geom.size = 1 ;

  /* if the output is 1x1 pixels, then there is no need to actually
   call im2col as it does not do anything
   */
  fullyConnectedMode = (output.geom.height == 1 && output.geom.width == 1) ;

  if (verbosity > 0) {
    mexPrintf("vl_nnconv: mode %s; %s\n", gpuMode?"gpu":"cpu", backMode?"backward":"forward") ;
    mexPrintf("vl_nnconv: stride: %d, pad: %d, numGroups: %d, bias: %d, fully connected: %d\n", stride, pad, numGroups, biasMode, fullyConnectedMode) ;
    feature_map_display(&data, "data") ;
    feature_map_display(&filters, "filters") ;
    if (biasMode) { feature_map_display(&biases, "biases") ; }
    if (backMode) {
      feature_map_display(&derOutput, "derOutput") ;
      feature_map_display(&derData, "derData") ;
      feature_map_display(&derFilters, "derFilters") ;
      if (biasMode) { feature_map_display(&derBiases, "derBiases") ; }
    } else {
      feature_map_display(&output, "output") ;
    }
    feature_map_display(&temp, "temp") ;
  }

  if (backMode) {
    if (derOutput.geom.height != temp.geom.height ||
        derOutput.geom.width != temp.geom.width ||
        derOutput.geom.depth != filters.geom.size ||
        derOutput.geom.size != data.geom.size)
    {
      mexErrMsgTxt("DEROUTPUT dimensions are incompatible with X and FILTERS.") ;
    }
  }

  if (numGroups * filters.geom.depth != data.geom.depth) {
    mexErrMsgTxt("The filter depth does not divide the image depth.") ;
  }

  if (filters.geom.size % numGroups != 0) {
    mexErrMsgTxt("The number of filter groups does not divide the total number of filters.") ;
  }

  if (data.geom.height + 2*pad < filters.geom.height || data.geom.width + 2*pad < filters.geom.width) {
    mexErrMsgTxt("FILTERS are larger than the DATA (including padding).") ;
  }

  if (filters.geom.height == 0 || filters.geom.width == 0 || filters.geom.depth == 0) {
    mexErrMsgTxt("A dimension of FILTERS is void.") ;
  }

  if (biasMode) {
    if (biases.geom.numElements != filters.geom.size) {
      mexErrMsgTxt("The number of elements of BIASES is not the same as the number of filters.") ;
    }
  }

  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */

  if (biasMode) {
    feature_map_init_with_geom_and_ones(&allOnes, gpuMode, allOnes.geom) ;
  }
  if (!backMode && computeOutput) {
    feature_map_init_with_geom(&output, gpuMode, output.geom, false) ;
  }
  if (!fullyConnectedMode) {
    feature_map_init_with_geom (&temp, gpuMode, temp.geom, false);
  }
  if (backMode && computeDerData) {
    feature_map_init_with_geom(&derData, gpuMode, derData.geom, fullyConnectedMode) ;
  }
  if (backMode && computeDerFilters) {
    feature_map_init_with_geom(&derFilters, gpuMode, derFilters.geom, true) ;
  }
  if (backMode && biasMode && computeDerBiases) {
    feature_map_init_with_geom(&derBiases, gpuMode, derBiases.geom, true) ;
  }

  for (int image = 0 ; image < data.geom.size ; ++image) {
    /*
     temp (phi(x)): m x k
     filters, derFilters: k x n (for one group of filters)
     derOutput (dzdy) : m x n (for one group of filters)
     res (y) : m x n (for one group of filters)
     */
    ptrdiff_t dataOffset = (data.geom.height*data.geom.width*data.geom.depth) * image ;
    ptrdiff_t outputOffset = (output.geom.height*output.geom.width*output.geom.depth) * image ;
    ptrdiff_t derDataOffset = dataOffset ;
    ptrdiff_t derOutputOffset = outputOffset ;
    ptrdiff_t m = temp.geom.height * temp.geom.width ; /* num output pixels */
    ptrdiff_t n = filters.geom.size/numGroups ; /* num filters per group */
    ptrdiff_t k = filters.geom.height*filters.geom.width*filters.geom.depth ; /* filter volume */
    char OP_N = 'n' ;
    char OP_T = 't' ;

    if (backMode) {
      /* ---------------------------------------------------------- */
      /*                                              Backward mode */
      /* ---------------------------------------------------------- */

      /* compute derFilters dz/dF */
      if (computeDerFilters) {
        if (!fullyConnectedMode) {
          if (gpuMode) {
#ifdef ENABLE_GPU
            im2col_gpu<float>((float const*)mxGPUGetDataReadOnly(data.gpuArray) + dataOffset,
                              data.geom.depth, data.geom.width, data.geom.height,
                              filters.geom.width, filters.geom.height,
                              stride, pad,
                              (float *)mxGPUGetData(temp.gpuArray)) ;
#else
            assert(false) ;
#endif
          } else {
            im2col_cpu<float>((float const*)mxGetData(data.array) + dataOffset,
                              data.geom.depth, data.geom.width, data.geom.height,
                              filters.geom.width, filters.geom.height,
                              stride, pad,
                              (float *)mxGetData(temp.array)) ;
          }
        }
        for (int g = 0 ; g < numGroups ; ++ g) {
          ptrdiff_t filterGrpOffset = k * n * g ;
          ptrdiff_t tempGrpOffset = m * k * g ;
          ptrdiff_t derOutputGrpOffset = m * n * g ;
          float alpha = 1 ;
          float beta = 1 ;
          if (gpuMode) {
#ifdef ENABLE_GPU
            cublasSgemm(handle,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        (int)k, (int)n, (int)m,
                        &alpha,
                        (float const*)mxGPUGetDataReadOnly(fullyConnectedMode ? data.gpuArray : temp.gpuArray)
                        + (fullyConnectedMode?dataOffset:0) + tempGrpOffset,
                        (int)m,
                        (float const*)mxGPUGetDataReadOnly(derOutput.gpuArray) + derOutputOffset + derOutputGrpOffset,
                        (int)m,
                        &beta,
                        (float*)mxGPUGetData(derFilters.gpuArray) + filterGrpOffset, (int)k) ;
#else
            assert(false) ;
#endif
          } else {
            sgemm(&OP_T, &OP_N,
                  &k, &n, &m,
                  &alpha,
                  (float*)mxGetData(fullyConnectedMode ? data.array : temp.array)
                  + (fullyConnectedMode?dataOffset:0) + tempGrpOffset, &m,
                  (float*)mxGetData(derOutput.array) + derOutputOffset + derOutputGrpOffset, &m,
                  &beta,
                  (float*)mxGetData(derFilters.array) + filterGrpOffset, &k) ;
          }
        }
      }

      /* compute derData dz/dbias */
      if (biasMode) {
        float alpha = 1 ;
        float beta = 1 ;
        ptrdiff_t q = filters.geom.size ;
        ptrdiff_t incx = 1 ;
        ptrdiff_t incy = 1 ;
        if (gpuMode) {
#ifdef ENABLE_GPU
          cublasSgemv(handle,
                      CUBLAS_OP_T,
                      (int)m, (int)q,
                      &alpha,
                      (float const*)mxGPUGetDataReadOnly(derOutput.gpuArray) + derOutputOffset, (int)m,
                      (float const*)mxGPUGetDataReadOnly(allOnes.gpuArray), (int)incx,
                      &beta,
                      (float*)mxGPUGetData(derBiases.gpuArray), (int)incy) ;
#else
          assert(false) ;
#endif
        } else {
          sgemv(&OP_T,
                &m, &q,
                &alpha,
                (float*)mxGetData(derOutput.array) + derOutputOffset, &m,
                (float*)mxGetData(allOnes.array), &incx,
                &beta,
                (float*)mxGetData(derBiases.array), &incy) ;
        }
      }

      /* compute derData dz/dx */
      if (computeDerData) {
        for (int g = 0 ; g < numGroups ; ++ g) {
          ptrdiff_t filterGrpOffset = k * n * g ;
          ptrdiff_t tempGrpOffset = m * k * g ;
          ptrdiff_t derOutputGrpOffset = m * n * g ;
          float alpha = 1 ;
          float beta = fullyConnectedMode ? 1 : 0 ;

          if (gpuMode) {
#ifdef ENABLE_GPU
            cublasSgemm(handle,
                        CUBLAS_OP_N, CUBLAS_OP_T,
                        (int)m, (int)k, (int)n,
                        &alpha,
                        (float const*)mxGPUGetDataReadOnly(derOutput.gpuArray) + derOutputOffset + derOutputGrpOffset, (int)m,
                        (float const*)mxGPUGetDataReadOnly(filters.gpuArray) + filterGrpOffset, (int)k,
                        &beta,
                        (float*)mxGPUGetData(fullyConnectedMode ? derData.gpuArray : temp.gpuArray)
                        + (fullyConnectedMode ? + derDataOffset : 0) + tempGrpOffset,
                        (int)m) ;
#else
            assert(false) ;
#endif
          } else {
            sgemm(&OP_N, &OP_T,
                  &m, &k, &n,
                  &alpha,
                  (float*)mxGetData(derOutput.array) + derOutputOffset + derOutputGrpOffset, &m,
                  (float*)mxGetData(filters.array) + filterGrpOffset, &k,
                  &beta,
                  (float*)mxGetData(fullyConnectedMode ? derData.array : temp.array)
                  + (fullyConnectedMode ? + derDataOffset : 0) + tempGrpOffset,
                  &m) ;
          }
        }
        if (!fullyConnectedMode) {
          if (gpuMode) {
#ifdef ENABLE_GPU
            col2im_gpu<float>((float*)mxGPUGetDataReadOnly(temp.gpuArray),
                              data.geom.depth, data.geom.width, data.geom.height,
                              filters.geom.width, filters.geom.height,
                              stride, pad,
                              (float*)mxGPUGetData(derData.gpuArray) + derDataOffset) ;
#else
            assert(false) ;
#endif
          } else {
            col2im_cpu<float>((float*)mxGetData(temp.array),
                              data.geom.depth, data.geom.width, data.geom.height,
                              filters.geom.width, filters.geom.height,
                              stride, pad,
                              (float*)mxGetData(derData.array) + derDataOffset) ;
          }
        }
      }
    } else {
      /* ---------------------------------------------------------- */
      /*                                               Forward mode */
      /* ---------------------------------------------------------- */
      if (computeOutput) {
        if (!fullyConnectedMode) {
          if (gpuMode) {
#ifdef ENABLE_GPU
            im2col_gpu<float>((float const*)mxGPUGetDataReadOnly(data.gpuArray) + dataOffset,
                              data.geom.depth, data.geom.width, data.geom.height,
                              filters.geom.width, filters.geom.height,
                              stride, pad,
                              (float *)mxGPUGetData(temp.gpuArray)) ;
#else
            assert(false) ;
#endif
          } else {
            im2col_cpu<float>((float const*)mxGetData(data.array) + dataOffset,
                              data.geom.depth, data.geom.width, data.geom.height,
                              filters.geom.width, filters.geom.height,
                              stride, pad,
                              (float *)mxGetData(temp.array)) ;
          }
        }
        for (int g = 0 ; g < numGroups ; ++ g) {
          ptrdiff_t filterGrpOffset = k * n * g ;
          ptrdiff_t tempGrpOffset = m * k * g ;
          ptrdiff_t outputGrpOffset = m * n * g  ;
          float alpha = 1 ;
          float beta = 0 ;
          if (gpuMode) {
#ifdef ENABLE_GPU
            cublasSgemm(handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        (int)m, (int)n, (int)k,
                        &alpha,
                        (float const*)mxGPUGetDataReadOnly(fullyConnectedMode ? data.gpuArray  : temp.gpuArray)
                        + (fullyConnectedMode?dataOffset:0) + tempGrpOffset, (int)m,
                        (float const*)mxGPUGetDataReadOnly(filters.gpuArray) + filterGrpOffset, (int)k,
                        &beta,
                        (float*)mxGPUGetData(output.gpuArray) + outputOffset + outputGrpOffset, (int)m) ;
#else
            assert(false) ;
#endif
          } else {
            sgemm(&OP_N, &OP_N,
                  &m, &n, &k,
                  &alpha,
                  (float*)mxGetData(fullyConnectedMode ? data.array : temp.array)
                  + (fullyConnectedMode?dataOffset:0) + tempGrpOffset, &m,
                  (float*)mxGetData(filters.array) + filterGrpOffset, &k,
                  &beta,
                  (float*)mxGetData(output.array) + outputOffset + outputGrpOffset, &m) ;
          }
        }
        if (biasMode) {
          float alpha = 1 ;
          float beta = 1 ;
          ptrdiff_t q = 1 ;
          if (gpuMode) {
#ifdef ENABLE_GPU
            cublasSgemm(handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        (int)m, (int)biases.geom.numElements, (int)q,
                        &alpha,
                        (float const*)mxGPUGetDataReadOnly(allOnes.gpuArray) , (int)m,
                        (float const*)mxGPUGetDataReadOnly(biases.gpuArray), (int)q,
                        &beta,
                        (float*)mxGPUGetData(output.gpuArray) + outputOffset, (int)m) ;
#else
            assert(false) ;
#endif
          } else {
            sgemm(&OP_N, &OP_N,
                  &m, &biases.geom.numElements, &q,
                  &alpha,
                  (float*)mxGetData(allOnes.array), &m,
                  (float*)mxGetData(biases.array), &q,
                  &beta,
                  (float*)mxGetData(output.array) + outputOffset, &m) ;
          }
        }
      }
    }
  }

  /* -------------------------------------------------------------- */
  /*                                                        Cleanup */
  /* -------------------------------------------------------------- */
#ifdef ENABLE_GPU
  if (gpuMode) {
    cublasDestroy(handle) ;
  }
#endif

  if (!fullyConnectedMode) { feature_map_deinit(&temp) ; }
  feature_map_deinit(&data) ;
  feature_map_deinit(&filters) ;
  if (biasMode) {
    feature_map_deinit(&biases) ;
    feature_map_deinit(&allOnes) ;
  }
  if (backMode) {
    out[OUT_RESULT] = feature_map_deinit_extracting_array(&derData) ;
    out[OUT_DERFILTERS] = feature_map_deinit_extracting_array(&derFilters) ;
    out[OUT_DERBIASES] = biasMode ? feature_map_deinit_extracting_array(&derBiases) : NULL ;
  } else {
    out[OUT_RESULT] = feature_map_deinit_extracting_array(&output) ;
  }
}
