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

#include <iostream>
#include <assert.h>

#include <blas.h>
#ifdef ENABLE_GPU
#include <cublas_v2.h>
#endif

/* option codes */
enum {
  opt_stride = 0,
  opt_pad,
  opt_verbose,
  opt_no_der_data,
  opt_no_der_filters,
  opt_no_der_biases,
} ;

/* options */
vlmxOption  options [] = {
  {"Stride",           1,   opt_stride             },
  {"Pad",              1,   opt_pad                },
  {"Verbose",          0,   opt_verbose            },
  {"NoDerData",        0,   opt_no_der_data        },
  {"NoDerFilters",     0,   opt_no_der_filters     },
  {"NoDerBiases",      0,   opt_no_der_biases      },
  {0,                  0,   0                      }
} ;

extern "C" bool mxUnshareArray(mxArray *array_ptr, bool noDeepCopy);

/* ---------------------------------------------------------------- */
/*                                                 Helper functions */
/* ---------------------------------------------------------------- */

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

int
compare_geom(PackedDataGeometry * a, PackedDataGeometry *b)
{
  return
    (a->height == b->height) &&
    (a->width  == b->width) &&
    (a->depth  == b->depth) &&
    (a->size   == b->size) ;
}

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
  geom->width = width;
  geom->depth = depth;
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
  mexPrintf("vl_nnconv: %s: %d x %d x %d x %d [%.1f MB]\n",
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
    if (!persistent) {
      /* if not persistent, create a GPU array */
      mwSize dimensions [4] = {geom.height, geom.width, geom.depth, geom.size} ;
      map->gpuArray = mxGPUCreateGPUArray
      (4, dimensions, mxSINGLE_CLASS, mxREAL,
       (initialize && value == 0) ? MX_GPU_INITIALIZE_VALUES : MX_GPU_DO_NOT_INITIALIZE) ;
      map->mode = matlabGpuArray ;
      map->memory = (float*) mxGPUGetData(map->gpuArray) ;
      if (initialize && value != 0) {
        cudaMemcpy(map->memory, memory, map->memorySize, cudaMemcpyHostToDevice) ;
      }
    } else {
      /* if persistent, use CUDA to allocate the memory (MATLAB does not have persistent GPU arrays) */
      map->mode = cudaMallocMemory ;
      cudaMalloc((void**)&map->memory, map->memorySize) ;
      cudaMemcpy(map->memory, memory, map->memorySize, cudaMemcpyHostToDevice) ;
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

/* ---------------------------------------------------------------- */
/*                                                            Cache */
/* ---------------------------------------------------------------- */

bool persistentDataInitialized = false ;
PackedData temp ;
PackedData allOnes ;

void atExit()
{
  packed_data_deinit (&temp)  ;
  packed_data_deinit (&allOnes)  ;
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
  PackedData data ;
  PackedData filters ;
  PackedData biases ;
  PackedData derOutput ;

  /* outputs */
  PackedData output ;
  PackedData derData  ;
  PackedData derFilters ;
  PackedData derBiases ;

  PackedDataGeometry outputGeom ;
  PackedDataGeometry derDataGeom  ;
  PackedDataGeometry derFiltersGeom ;
  PackedDataGeometry derBiasesGeom ;
  PackedDataGeometry tempGeom ;
  PackedDataGeometry allOnesGeom ;

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

  packed_data_init_empty(&data) ;
  packed_data_init_empty(&filters) ;
  packed_data_init_empty(&biases) ;
  packed_data_init_empty(&derOutput) ;
  packed_data_init_empty(&output) ;
  packed_data_init_empty(&derData) ;
  packed_data_init_empty(&derFilters) ;
  packed_data_init_empty(&derBiases) ;
  if (!persistentDataInitialized) {
    persistentDataInitialized = true ;
    packed_data_init_empty(&temp) ;
    packed_data_init_empty(&allOnes) ;
  }

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  mexAtExit(atExit) ;

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

    case opt_no_der_data :
      computeDerData = VL_FALSE ;
      break ;

    case opt_no_der_filters :
      computeDerFilters = VL_FALSE ;
      break ;

    case opt_no_der_biases :
      computeDerBiases = VL_FALSE ;
      break ;

    default: break ;
    }
  }

  packed_data_init_with_array (&data, gpuMode, in[IN_DATA]) ;
  packed_data_init_with_array (&filters, gpuMode, in[IN_FILTERS]) ;
  if (biasMode) { packed_data_init_with_array(&biases, gpuMode, in[IN_BIASES]) ; }
  if (backMode) { packed_data_init_with_array(&derOutput, gpuMode, in[IN_DEROUTPUT]) ; }

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

  packed_data_geom_init(&outputGeom,
                        mxSINGLE_CLASS,
                        (data.geom.height + 2*pad - filters.geom.height)/stride + 1,
                        (data.geom.width + 2*pad - filters.geom.width)/stride + 1,
                        filters.geom.size,
                        data.geom.size) ;

  /* grouped filters */
  numGroups = data.geom.depth / filters.geom.depth ;

  /* if the output is 1x1 pixels, then there is no need to actually
   call im2col as it does not do anything
   */
  fullyConnectedMode = (outputGeom.height == 1 && outputGeom.width == 1 && numGroups == 1) ;

  derDataGeom = data.geom ;
  derFiltersGeom = filters.geom ;
  if (biasMode) {
    if (fullyConnectedMode) {
      packed_data_geom_init (&allOnesGeom, mxSINGLE_CLASS,
                             1, 1,
                             1, data.geom.size) ;
    } else {
      packed_data_geom_init (&allOnesGeom, mxSINGLE_CLASS,
                             outputGeom.height,
                             outputGeom.width,
                             1, 1) ;
    }
    derBiasesGeom = biases.geom ;
  } else {
    packed_data_geom_init (&allOnesGeom, mxSINGLE_CLASS,
                           0, 0, 0, 0) ;
  }

  packed_data_geom_init
  (&tempGeom,
   mxSINGLE_CLASS,
   outputGeom.height,
   outputGeom.width,
   filters.geom.height*filters.geom.width*filters.geom.depth*numGroups,
   1) ;

  if (verbosity > 0) {
    mexPrintf("vl_nnconv: mode %s; %s\n", gpuMode?"gpu":"cpu", backMode?"backward":"forward") ;
    mexPrintf("vl_nnconv: stride: %d, pad: %d, numGroups: %d, bias: %d, fully connected: %d\n", stride, pad, numGroups, biasMode, fullyConnectedMode) ;
    packed_data_geom_display(&data.geom, "data") ;
    packed_data_geom_display(&filters.geom, "filters") ;
    if (biasMode) { packed_data_geom_display(&biases.geom, "biases") ; }
    if (backMode) {
      packed_data_geom_display(&derOutput.geom, "derOutput") ;
      packed_data_geom_display(&derDataGeom, "derData") ;
      packed_data_geom_display(&derFiltersGeom, "derFilters") ;
      if (biasMode) { packed_data_geom_display(&derBiasesGeom, "derBiases") ; }
    } else {
      packed_data_geom_display(&outputGeom, "output") ;
    }
    packed_data_geom_display(&tempGeom, "temp") ;
    packed_data_geom_display(&temp.geom, "temp (cached)") ;
    packed_data_geom_display(&allOnesGeom, "allOnes") ;
    packed_data_geom_display(&allOnes.geom, "allOnes (cached)") ;
  }

  if (backMode) {
    if (derOutput.geom.height != tempGeom.height ||
        derOutput.geom.width != tempGeom.width ||
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

  /* auxiliary buffers */
  if (biasMode) {
    if (allOnes.memorySize < allOnesGeom.numElements * sizeof(float)) {
      packed_data_deinit (&allOnes) ;
      packed_data_init_with_geom (&allOnes, gpuMode, allOnesGeom, true, true, 1.0f) ;
    }
  }
  if (!fullyConnectedMode) {
    if (temp.memorySize < tempGeom.numElements * sizeof(float)) {
      packed_data_deinit (&temp) ;
      packed_data_init_with_geom (&temp, gpuMode, tempGeom, true, false, 0);
    }
  }
  if (!backMode && computeOutput) {
    packed_data_init_with_geom(&output, gpuMode, outputGeom, false, false, 0) ;
  }
  if (backMode && computeDerData) {
    packed_data_init_with_geom(&derData, gpuMode, derDataGeom, false, false, 0) ;
  }
  if (backMode && computeDerFilters) {
    packed_data_init_with_geom(&derFilters, gpuMode, derFiltersGeom, false, false, 0) ;
  }
  if (backMode && computeDerBiases) {
    packed_data_init_with_geom(&derBiases, gpuMode, derBiasesGeom, false, false, 0) ;
  }

  if (fullyConnectedMode) {
    float alpha = 1 ;
    float beta = 0 ;
    ptrdiff_t incx = 1 ;
    ptrdiff_t incy = 1 ;
    char OP_N = 'n' ;
    char OP_T = 't' ;
    ptrdiff_t filtersVolume = filters.geom.height*filters.geom.width*filters.geom.depth ;

    /* especially optimized */
    if (!backMode) {
      if (data.geom.size == 1) {
        /* one image in the stack */
        sgemv(&OP_T,
              &filtersVolume, &filters.geom.size,
              &alpha,
              filters.memory, &filtersVolume,
              data.memory, &incx,
              &beta,
              output.memory, &incy) ;
      } else {
        /* multiple images in the stack */
        sgemm(&OP_T, &OP_N,
              &filters.geom.size, &data.geom.size, &filtersVolume,
              &alpha,
              filters.memory, &filtersVolume,
              data.memory, &filtersVolume,
              &beta,
              output.memory, &filters.geom.size) ;
      }
      if (biasMode) {
        float beta = 1 ;
        ptrdiff_t q = 1 ;
        sgemm(&OP_N, &OP_N,
              &filters.geom.size, &data.geom.size, &q,
              &alpha,
              biases.memory, &filters.geom.size,
              allOnes.memory, &q,
              &beta,
              output.memory, &filters.geom.size) ;
      }
    } else {
      /* back mode */
      if (computeDerFilters) {
        sgemm(&OP_N, &OP_T,
              &filtersVolume, &filters.geom.size, &data.geom.size,
              &alpha,
              data.memory, &filtersVolume,
              derOutput.memory, &filters.geom.size,
              &beta,
              derFilters.memory, &filtersVolume) ;
      }
      if (computeDerBiases & biasMode) {
        ptrdiff_t q = 1 ;
        sgemm(&OP_N, &OP_T,
              &q, &filters.geom.size, &data.geom.size,
              &alpha,
              allOnes.memory, &q,
              derOutput.memory, &filters.geom.size,
              &beta,
              derBiases.memory, &q) ;
      }
      if (computeDerData) {
        sgemm(&OP_N, &OP_N,
              &filtersVolume, &data.geom.size, &filters.geom.size,
              &alpha,
              filters.memory, &filtersVolume,
              derOutput.memory, &filters.geom.size,
              &beta,
              derData.memory, &filtersVolume) ;
      }
    }
  } else {
    /* not fully connected */
    for (int image = 0 ; image < data.geom.size ; ++image) {
      /*
        temp (phi(x)): m x k
        filters, derFilters: k x n (for one group of filters)
        derOutput (dzdy) : m x n (for one group of filters)
        res (y) : m x n (for one group of filters)
      */
      ptrdiff_t dataOffset = (data.geom.height*data.geom.width*data.geom.depth) * image ;
      ptrdiff_t outputOffset = (output.geom.height*output.geom.width*output.geom.depth) * image ;
      ptrdiff_t derDataOffset = (derData.geom.height*derData.geom.width*derData.geom.depth) * image ;
      ptrdiff_t derOutputOffset = (derOutput.geom.height*derOutput.geom.width*derOutput.geom.depth) * image ;
      ptrdiff_t m = tempGeom.height * tempGeom.width ; /* num output pixels */
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
              im2col_gpu<float>(data.memory + dataOffset,
                                data.geom.depth, data.geom.width, data.geom.height,
                                filters.geom.width, filters.geom.height,
                                stride, pad,
                                temp.memory) ;
#else
              assert(false) ;
#endif
            } else {
              im2col_cpu<float>(data.memory + dataOffset,
                                data.geom.depth, data.geom.width, data.geom.height,
                                filters.geom.width, filters.geom.height,
                                stride, pad,
                                temp.memory) ;
            }
          }
          for (int g = 0 ; g < numGroups ; ++ g) {
            ptrdiff_t filterGrpOffset = k * n * g ;
            ptrdiff_t tempGrpOffset = m * k * g ;
            ptrdiff_t derOutputGrpOffset = m * n * g ;
            float alpha = 1 ;
            float beta = (image > 0)  ;
            if (gpuMode) {
#ifdef ENABLE_GPU
              cublasSgemm(handle,
                          CUBLAS_OP_T, CUBLAS_OP_N,
                          (int)k, (int)n, (int)m,
                          &alpha,
                          (fullyConnectedMode ? data.memory : temp.memory)
                          + (fullyConnectedMode?dataOffset:0) + tempGrpOffset,
                          (int)m,
                          derOutput.memory + derOutputOffset + derOutputGrpOffset,
                          (int)m,
                          &beta,
                          derFilters.memory + filterGrpOffset, (int)k) ;
#else
              assert(false) ;
#endif
            } else {
              sgemm(&OP_T, &OP_N,
                    &k, &n, &m,
                    &alpha,
                    (fullyConnectedMode ? data.memory : temp.memory)
                    + (fullyConnectedMode?dataOffset:0) + tempGrpOffset, &m,
                    derOutput.memory + derOutputOffset + derOutputGrpOffset, &m,
                    &beta,
                    derFilters.memory + filterGrpOffset, &k) ;
            }
          }
        }

        /* compute derData dz/dbias */
        if (computeDerBiases & biasMode) {
          float alpha = 1 ;
          float beta = (image > 0) ;
          ptrdiff_t q = filters.geom.size ;
          ptrdiff_t incx = 1 ;
          ptrdiff_t incy = 1 ;
          if (gpuMode) {
#ifdef ENABLE_GPU
            cublasSgemv(handle,
                        CUBLAS_OP_T,
                        (int)m, (int)q,
                        &alpha,
                        derOutput.memory + derOutputOffset, (int)m,
                        allOnes.memory, (int)incx,
                        &beta,
                        derBiases.memory, (int)incy) ;
#else
            assert(false) ;
#endif
          } else {
            sgemv(&OP_T,
                  &m, &q,
                  &alpha,
                  derOutput.memory + derOutputOffset, &m,
                  allOnes.memory, &incx,
                  &beta,
                  derBiases.memory, &incy) ;
          }
        }

        /* compute derData dz/dx */
        if (computeDerData) {
          for (int g = 0 ; g < numGroups ; ++ g) {
            ptrdiff_t filterGrpOffset = k * n * g ;
            ptrdiff_t tempGrpOffset = m * k * g ;
            ptrdiff_t derOutputGrpOffset = m * n * g ;
            float alpha = 1 ;
            float beta = fullyConnectedMode ? (g > 0) : 0 ;

            if (gpuMode) {
#ifdef ENABLE_GPU
              cublasSgemm(handle,
                          CUBLAS_OP_N, CUBLAS_OP_T,
                          (int)m, (int)k, (int)n,
                          &alpha,
                          derOutput.memory + derOutputOffset + derOutputGrpOffset, (int)m,
                          filters.memory + filterGrpOffset, (int)k,
                          &beta,
                          (fullyConnectedMode ? derData.memory : temp.memory)
                          + (fullyConnectedMode ? + derDataOffset : 0) + tempGrpOffset,
                          (int)m) ;
#else
              assert(false) ;
#endif
            } else {
              sgemm(&OP_N, &OP_T,
                    &m, &k, &n,
                    &alpha,
                    derOutput.memory + derOutputOffset + derOutputGrpOffset, &m,
                    filters.memory + filterGrpOffset, &k,
                    &beta,
                    (fullyConnectedMode ? derData.memory : temp.memory)
                    + (fullyConnectedMode ? + derDataOffset : 0) + tempGrpOffset,
                    &m) ;
            }
          }
#if 1
          if (!fullyConnectedMode) {
            if (gpuMode) {
#ifdef ENABLE_GPU
              col2im_gpu<float>(temp.memory,
                                data.geom.depth, data.geom.width, data.geom.height,
                                filters.geom.width, filters.geom.height,
                                stride, pad,
                                derData.memory + derDataOffset) ;
#else
              assert(false) ;
#endif
            } else {
              col2im_cpu<float>(temp.memory,
                                data.geom.depth, data.geom.width, data.geom.height,
                                filters.geom.width, filters.geom.height,
                                stride, pad,
                                derData.memory + derDataOffset) ;
            }
          }
#endif
        }
      } else {
        /* ---------------------------------------------------------- */
        /*                                               Forward mode */
        /* ---------------------------------------------------------- */
        if (computeOutput) {
          if (gpuMode) {
#ifdef ENABLE_GPU
            im2col_gpu<float>(data.memory + dataOffset,
                              data.geom.depth, data.geom.width, data.geom.height,
                              filters.geom.width, filters.geom.height,
                              stride, pad,
                              temp.memory) ;
#else
            assert(false) ;
#endif
          } else {
            im2col_cpu<float>(data.memory + dataOffset,
                              data.geom.depth, data.geom.width, data.geom.height,
                              filters.geom.width, filters.geom.height,
                              stride, pad,
                              temp.memory) ;
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
                        (fullyConnectedMode ? data.memory : temp.memory)
                        + (fullyConnectedMode?dataOffset:0) + tempGrpOffset, (int)m,
                        filters.memory + filterGrpOffset, (int)k,
                        &beta,
                        output.memory + outputOffset + outputGrpOffset, (int)m) ;
#else
            assert(false) ;
#endif
          } else {
            sgemm(&OP_N, &OP_N,
                  &m, &n, &k,
                  &alpha,
                  (fullyConnectedMode ? data.memory : temp.memory)
                  + (fullyConnectedMode?dataOffset:0) + tempGrpOffset, &m,
                  filters.memory + filterGrpOffset, &k,
                  &beta,
                  output.memory + outputOffset + outputGrpOffset, &m) ;
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
                        allOnes.memory, (int)m,
                        biases.memory, (int)q,
                        &beta,
                        output.memory + outputOffset, (int)m) ;
#else
            assert(false) ;
#endif
          } else {
            sgemm(&OP_N, &OP_N,
                  &m, &biases.geom.numElements, &q,
                  &alpha,
                  allOnes.memory, &m,
                  biases.memory, &q,
                  &beta,
                  output.memory + outputOffset, &m) ;
          }
        }
      }
    }
  }

  /* -------------------------------------------------------------- */
  /*                                                        Cleanup */
  /* -------------------------------------------------------------- */

#ifdef ENABLE_GPU
  if (gpuMode) { cublasDestroy(handle) ; }
#endif

  packed_data_deinit(&data) ;
  packed_data_deinit(&filters) ;
  if (biasMode) { packed_data_deinit(&biases) ; }
  if (backMode) {
    out[OUT_RESULT] = (computeDerData) ? packed_data_deinit_extracting_array(&derData) : mxCreateDoubleMatrix(0,0,mxREAL) ;
    out[OUT_DERFILTERS] =(computeDerFilters)? packed_data_deinit_extracting_array(&derFilters) : mxCreateDoubleMatrix(0,0,mxREAL) ;
    out[OUT_DERBIASES] = (computeDerBiases & biasMode) ? packed_data_deinit_extracting_array(&derBiases) : mxCreateDoubleMatrix(0,0,mxREAL) ;
  } else {
    out[OUT_RESULT] = packed_data_deinit_extracting_array(&output) ;
  }
}
