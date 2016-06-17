/** @file vl_tflowmex.cu
 ** @brief MEX internals of vl_tflowmex.m.
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2016 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "bits/mexutils.h"
#include "bits/datamex.hpp"

#include "bits/data.hpp"
#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif

#include "bits/impl/tinythread.h"
#include "bits/impl/blashelper.hpp"

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/shm.h>
#include <sys/un.h>
#include <sys/socket.h>

#include <memory>
#include <vector>
#include <algorithm>
#include <sstream>

/**
 \file vl_tflowmex.cu
 
 The `vl_tflowmex` utility implements an efficient mechanism to exchange
 tensor data between different MATLAB processes. Presently, it is
 limited to processes running on the same host, but future extensions
 can integrate networked environments. Even limited to a single
 host, this functionality is important because MATLAB multiple GPU
 support uses different processess for different GPUs.
 
 The key idea is to implement a reduction tree, in which each MATLAB
 process is connected to a parent and a number of children. When a tensor
 needs to be accumulated, a node receives copies form the children,
 sums them with its local copy, and sends the result to the parent.
 Eventually, the data flow reaches the root of the tree and the accumulated
 tensor is sent back towards the leaves. This communication mechanism
 is designed to reduce the amount of data transfers from O(n^2)
 for the trivial n-to-n communication of tensor copies to O(n).
 
 A second strategy used to significantly improve the speed is to allow
 the transfer of tensor data to proceed in the background, while MATLAB is busy
 running the rest of the network. This is achieved by isolating
 all communications in a supervisory thread.

 # Notable facts
 
 * Communications between thread uses UNIX-domain sockets (extensible
   to INet sockets in the future). These are used to send lightweight
   cohordination messages.
 
 * Data passing on local machines uses a shared memory map between
   processes. The shared memory contains a copy of each tensor for each
   process. GPU tensors may either be allocated internally
   by `vl_tflowmex` (in which case MATLAB may forget them)
   or may remember pointers to MATLAB's memory (inplace). 
   The latter is slightly unsafe, but much faster as it saves several copies.
   In any case, `vl_tflowmex` allocates a GPU buffer as large as
   the largest tensor as scratch space (and for direct GPU communication).

 * The supervisory and main threads collaborate through lock-less
   synchronization for speed. This is possible because at any point in time
   each tensor is managed by only one thread depending on its state.
   Thus a tensor moves from one thread to the other simply by swapping
   its state. There is, however, a condition variable to allow the
   main thread to wait for the supervisory thread when needed.

 * The supervisory thread waits by calling `poll()` on a number of sockets.
   However, sometimes the main thread needs to signal the supervisor too.
   This is realized by having a dummy `pipe()` between the two
   threads.

 **/

/* ---------------------------------------------------------------- */
/*                                                          Globals */
/* ---------------------------------------------------------------- */

enum {
  IN_COMMAND, IN_END
} ;

enum {
  OUT_RESULT, OUT_END
} ;

/* option codes */
enum {
  opt_inplace = 0,
  opt_verbose,
} ;

/* options */
VLMXOption  options [] = {
  {"InPlace",               0,   opt_inplace               },
  {"Verbose",               0,   opt_verbose               },
  {0,                       0,   0                         }
} ;

int verbosity = 0 ;
vl::MexContext context ;

class SharedTensorDescriptor ;
class SharedTensorSpace ;
class ProcessPool ;

/* ---------------------------------------------------------------- */
/*                                                          Utility */
/* ---------------------------------------------------------------- */

static VLMXErrorCode vlmxParseDataType(vl::DataType & dataType, mxArray const * arg)
{
  if (vlmxCompareToStringI(arg, "double") == 0) {
    dataType = vl::VLDT_Double ;
    return VLMXE_Success ;
  } else if (vlmxCompareToStringI(arg, "single") == 0) {
    dataType = vl::VLDT_Float ;
    return VLMXE_Success ;
  } else {
    return VLMXE_IllegalArgument ;
  }
}

static VLMXErrorCode vlmxParseDeviceType(vl::DeviceType & deviceType, mxArray const * arg)
{
  if (vlmxCompareToStringI(arg, "cpu") == 0) {
    deviceType = vl::VLDT_CPU ;
    return VLMXE_Success ;
  } else if (vlmxCompareToStringI(arg, "gpu") == 0) {
    deviceType = vl::VLDT_GPU ;
    return VLMXE_Success ;
  } else {
    return VLMXE_IllegalArgument ;
  }
}

static VLMXErrorCode vlmxParseString(std::string & name, mxArray const * arg)
{
  char str [256] ;
  if (!vlmxIsString(arg, -1)) {
    return VLMXE_IllegalArgument ;
  }
  mxGetString(arg, str, sizeof(str)) ;
  name = str ;
  return VLMXE_Success ;
}

static VLMXErrorCode vlmxParseTensorShape(vl::TensorShape & shape, mxArray const * arg)
{
  size_t dimensions [32] ;
  if (!vlmxIsVector(arg, -1) || !vlmxIsPlain(arg)) {
    return VLMXE_IllegalArgument ;
  }
  int nd = mxGetNumberOfElements(arg) ;
  for (int k = 0 ; k < nd ; ++k) { dimensions[k] = (size_t)mxGetPr(arg)[k] ; }
  shape.setDimensions(dimensions, nd) ;
  return VLMXE_Success ;
}

/* ---------------------------------------------------------------- */
/*                                                           Logger */
/* ---------------------------------------------------------------- */

namespace vl {
  class Logger
  {
  public:
    Logger() ;
    ~Logger() ;
    std::ostringstream & getStream() ;
  protected:
    std::ostringstream stringStream ;
  private:
    // Disable
    Logger(const Logger&) ;
    Logger& operator= (const Logger&) ;
  } ;
}

vl::Logger::Logger()
{ }

vl::Logger::~Logger()
{
  printf("%s\n", stringStream.str().c_str()) ;
  //fflush(stdout) ;
}

std::ostringstream &
vl::Logger::getStream()
{
  return stringStream ;
}

#define LOGERROR \
vl::Logger().getStream() \
<<"[error]"<<__func__<<"::lab "<<lab<<"::"

#define LOG(level) \
if (verbosity < level) { } \
else vl::Logger().getStream() \
<<"[info] "<<__func__<<"::lab "<<lab<<"::"

/* ---------------------------------------------------------------- */
/*                                           SharedTensorDescriptor */
/* ---------------------------------------------------------------- */

#pragma mark -

// Describe one of the shared tensors: shape, data type,
// and device type.
class SharedTensorDescriptor
{
public:
  SharedTensorDescriptor() ;
  ~SharedTensorDescriptor() ;

  void init(vl::DeviceType deviceType,
            vl::DataType dataType,
            vl::TensorShape const & shape) ;
  void finalize() ;
  size_t getSizeInBytes() const ;
  SharedTensorDescriptor & operator=(SharedTensorDescriptor const & tensor) ;

  vl::ErrorCode load(std::istream & is) ;
  vl::ErrorCode save(std::ostream & os) ;

  // Data.
  vl::DeviceType deviceType ;
  vl::DataType dataType ;
  vl::TensorShape shape ;
} ;

SharedTensorDescriptor::SharedTensorDescriptor()
{ }

SharedTensorDescriptor::~SharedTensorDescriptor()
{
  finalize() ;
}

SharedTensorDescriptor &
SharedTensorDescriptor::operator=(SharedTensorDescriptor const & tensor)
{
  deviceType = tensor.deviceType ;
  dataType = tensor.dataType ;
  shape = tensor.shape ;
  return *this ;
}

void SharedTensorDescriptor::init(vl::DeviceType newDeviceType,
                                  vl::DataType newDataType,
                                  vl::TensorShape const & newShape)
{
  assert(newDeviceType == vl::VLDT_CPU || newDeviceType == vl::VLDT_GPU) ;
  assert(newDataType == vl::VLDT_Float || newDataType == vl::VLDT_Double) ;
  deviceType = newDeviceType ;
  dataType = newDataType ;
  shape = newShape ;
}

void SharedTensorDescriptor::finalize()
{ }

size_t SharedTensorDescriptor::getSizeInBytes() const
{
  return shape.getNumElements() * getDataTypeSizeInBytes(dataType) ;
}

/* ---------------------------------------------------------------- */
/*                                                SharedTensorSpace */
/* ---------------------------------------------------------------- */

#pragma mark -

// SharedTensorSpace holds a list of tensors that can be accumulated
// between different processes.
//
// It encapsualtes in particular: the shared memory map,
// the GPU dispatch buffer, and, possibly, for non-inplace operations
// and GPU arrays, a copy of the GPU data.
//
// This class is not thread safe, so the MATLAB and flow supervisor thread
// must properly syncrhonize in accessing it.

class SharedTensorSpace
{
public:
  SharedTensorSpace() ;
  ~SharedTensorSpace() ;

  vl::ErrorCode mexInit(mxArray const *mexDescriptor) ;
  void finalize() ;
  vl::ErrorCode attach(int lab, int numLabs) ;
  vl::ErrorCode attachPeer(int lab) ;

  void mexPrint() const ;

private:
  bool initialized ;
  int lab ;
  int numLabs ;

  enum SharedTensorState {
    ready,
    accumulateChildren,
    waitParent,
    waitChildren,
  } state ;

  // This class represents an instance of a shared tensor. It contain
  // its state@transaction pair and information on its memory location.
  struct SharedTensorInstance
  {
    std::string name ;
    SharedTensorDescriptor descriptor ;
    SharedTensorState state ;
    size_t transaction ;
    int numChildrenToAccumulate ;
    size_t memoryMapOffset ;
    void * cpuMemory ;
    void * gpuMemory ;
    bool gpuMemoryIsOwned ;
#if ENABLE_GPU
    cudaEvent_t gpuEvent ;
    bool gpuEventIsInitialized ;
#endif
    bool operator==(std::string const & theName) { return name == theName ; }
    SharedTensorInstance()
    : state(ready), transaction(0),
      cpuMemory(NULL), gpuMemory(NULL), gpuMemoryIsOwned(false)
#if ENABLE_GPU
    , gpuEvent(0), gpuEventIsInitialized(false)
#endif
    { }
  } ;
  typedef std::vector<SharedTensorInstance> tensors_t ;
  tensors_t tensors ;

  struct SharedTensorPeerInstance
  {
    int lab ;
    SharedTensorState state ;
    size_t transaction ;
    void *mappedCpuMemory ;
    void *mappedGpuMemory ;
    bool accumulated ;
    bool operator==(int theLab) { return lab == theLab ; }
    SharedTensorPeerInstance()
    : lab(-1), state(ready), transaction(0), mappedCpuMemory(NULL), mappedGpuMemory(NULL), accumulated(false) { }
  } ;
  typedef std::vector<std::vector<SharedTensorPeerInstance> > peerTensors_t ;
  peerTensors_t peerTensors ;
  SharedTensorPeerInstance & getPeerTensor(int tensorIndex, int lab) ;

  // Shared CPU memory
  void * memoryMap ;
  int memoryMapSize ;
  int memoryMapLabStride ;
  std::string memoryMapName ;
  int memoryMapFD ;

  // Additional GPU memory
  void * gpuDispatchMemory ;
  int gpuDevice ;

#if ENABLE_GPU
  // Todo: one for each mapped peer dispatch memory
  cudaIpcMemHandle_t gpuMemoryHandle ;
  cudaStream_t gpuHelperStream ;
  cudaEvent_t gpuHelperEvent ;
  bool gpuHelperStreamInitialized ;
  bool gpuHelperEventInitialized ;
#endif

  friend class ProcessPool ;
} ;

SharedTensorSpace::SharedTensorSpace()
  : initialized(false), memoryMapFD(-1), memoryMap(NULL), gpuDevice(-1),
    gpuDispatchMemory(NULL)
#if ENABLE_GPU
,   gpuHelperStream(0),
    gpuHelperStreamInitialized(false),
    gpuHelperEventInitialized(false)
#endif
{ }

SharedTensorSpace::~SharedTensorSpace()
{
  finalize() ;
}

// This function initializes the SharedTensorSpace using
// a MATLAB cell array as descriptor for the space content.
// It can throw a MEX error, so it must be called from
// the MATLAB thread.

vl::ErrorCode SharedTensorSpace::mexInit(mxArray const *descriptor)
{
  assert(descriptor) ;

  if (initialized) {
    mexErrMsgTxt("Already initialized. Use 'reset' to clear.") ;
  }

  lab = -1 ;
  numLabs = 0 ;
  memoryMapName = "" ;
  memoryMapSize = 0 ;
  memoryMapLabStride = 0 ;

  // Parse tensor list
  if (!mxIsCell(descriptor)) {
    mexErrMsgTxt("DESCRIPTOR is not a cell array.") ;
  }
  if (mxGetNumberOfDimensions(descriptor) != 2) {
    mexErrMsgTxt("DESCRIPTOR does not have two dimensions.") ;
  }
  if (mxGetN(descriptor) != 3 &&
      mxGetN(descriptor) != 4) {
    mexErrMsgTxt("DESCRIPTOR does not have three or four columns.") ;
  }

  size_t numTensors = mxGetM(descriptor) ;
  size_t offset = 0 ;
  size_t const alignFactor = 16 ;
  bool useGPU = false ;

  for (int i = 0 ; i < numTensors ; ++i) {
    VLMXErrorCode error ;
    vl::DeviceType deviceType = vl::VLDT_CPU ;
    vl::DataType dataType ;
    vl::TensorShape shape ;
    std::string name ;

    error = vlmxParseDataType(dataType, mxGetCell(descriptor, 0*numTensors + i)) ;
    if (error != VLMXE_Success) {
      vlmxError(error, "DESCRIPTOR{%d,1} is not a valid data type.", i+1) ;
    }

    error = vlmxParseTensorShape(shape, mxGetCell(descriptor, 1*numTensors + i)) ;
    if (error != VLMXE_Success) {
      vlmxError(error, "DESCRIPTOR{%d,2} is not a valid tensor shape.", i+1) ;
    }

    error = vlmxParseString(name, mxGetCell(descriptor, 2*numTensors + i)) ;
    if (error != VLMXE_Success) {
      vlmxError(error, "DESCRIPTOR{%d,3} is not a valid tensor name.", i+1) ;
    }

    if (mxGetN(descriptor) == 4) {
      error = vlmxParseDeviceType(deviceType, mxGetCell(descriptor, 3*numTensors + i)) ;
      if (error != VLMXE_Success) {
        vlmxError(error, "DESCRIPTOR{%d,4} is not a valid device type name.", i+1) ;
      }
    }

    if (deviceType == vl::VLDT_GPU) {
#if not defined(ENABLE_GPU)
      vlmxError(VLMXE_IllegalArgument, "GPU support not compiled.") ;
#endif
      useGPU = true ;
    }

    // Add the new tensor to the table.
    {
      SharedTensorInstance tensor ;
      tensor.name = name ;
      tensor.descriptor.init(deviceType, dataType, shape) ;
      tensor.memoryMapOffset = offset ;
      tensors.push_back(tensor) ;

      offset +=
      vl::divideAndRoundUp(tensor.descriptor.getSizeInBytes(), alignFactor) * alignFactor ;
    }
  }

  // Size of the memory allocated for one lab (with a copy of all tensors).
  memoryMapName = "/mcn-shared-memory" ;
  size_t const pageSize = getpagesize() ;
  memoryMapLabStride = vl::divideAndRoundUp(offset, pageSize) * pageSize ;
  memoryMapSize = 0 ;

#if ENABLE_GPU
  if (useGPU) {
    cudaGetDevice(&gpuDevice) ; // to inform thread
    LOG(2) << "current CUDA device: " << gpuDevice ;
  }
#endif

  initialized = true ;
  return vl::VLE_Success ;
}

// Get the peer tensor corresponding to a given
// tensor and process index.

SharedTensorSpace::SharedTensorPeerInstance &
SharedTensorSpace::getPeerTensor(int tensorIndex, int lab)
{
  std::vector<SharedTensorPeerInstance>::iterator PT
  = std::find(peerTensors[tensorIndex].begin(), peerTensors[tensorIndex].end(), lab) ;
  assert(PT != peerTensors[tensorIndex].end()) ;
  return *PT ;
}

/// Attach the shared space. This allocates the shared memory map
/// for inter-process data transfers containing all tensors,
/// and the GPU dispatch memory.

vl::ErrorCode SharedTensorSpace::attach(int lab, int numLabs)
{
  int error ;
  this->lab = lab ;
  this->numLabs = numLabs ;

  // The root lab deletes a pre-existing memory object, if any.
  if (lab == 0) {
    error = shm_unlink(memoryMapName.c_str()) ;
    if (error == -1) {
      switch (errno) {
        case ENOENT:
          // Fine, there wasn't such a memory map anyways.
          break ;

        default:
          LOGERROR
          << "could not delete the stale memory map '"
          << memoryMapName.c_str()
          << "' because '" << strerror(errno) << '\'' ;
          return vl::VLE_Unknown ;
      }
    }
  }

  // Open/create the shared memory file descriptor.
  memoryMapSize = memoryMapLabStride * numLabs ;
  memoryMapFD = shm_open(memoryMapName.c_str(),
                         (lab == 0 ? O_CREAT:0)| O_RDWR, S_IRUSR | S_IWUSR) ;
  if (memoryMapFD == -1) {
    LOGERROR << "shm_open() failed because " << strerror(errno) ;
    return vl::VLE_Unknown ;
  }

  // The root process set the size of the shared memory.
  if (lab == 0) {
    if (ftruncate(memoryMapFD, memoryMapSize) == -1) {
      LOGERROR << "truncate failed because " << strerror(errno) ;
      return vl::VLE_OutOfMemory ;
    }
  }

  // Map the memory.
  memoryMap = mmap(0, memoryMapSize,
                   PROT_READ | PROT_WRITE, MAP_SHARED,
                   memoryMapFD, 0) ;
  if (memoryMap == MAP_FAILED) {
    LOGERROR << "mmap failed because " << strerror(errno) ;
    memoryMap = NULL ;
    return vl::VLE_Unknown ;
  }

  // Associate memory to tensors.
#if ENABLE_GPU
  size_t maxGPUTensorSize = 0 ;
#endif
  for (int t = 0 ; t < tensors.size() ; ++t) {
    tensors[t].cpuMemory = (char*)memoryMap
    + tensors[t].memoryMapOffset
    + lab * memoryMapLabStride ;
#if ENABLE_GPU
    if (tensors[t].descriptor.deviceType == vl::VLDT_GPU) {
      // Lazy allocation (to allow inplace operations).
      tensors[t].gpuMemory = NULL ;
      tensors[t].gpuMemoryIsOwned = false ;
      maxGPUTensorSize = std::max(maxGPUTensorSize,
                                  tensors[t].descriptor.getSizeInBytes()) ;

      cudaError_t cerror = cudaEventCreate(&tensors[t].gpuEvent) ;
      if (cerror != cudaSuccess) {
        LOGERROR
          << "CUDA could not create an event because '"
          << cudaGetErrorString(cerror) << '\'' ;
        return vl::VLE_Cuda ;
      }
      tensors[t].gpuEventIsInitialized = true ;
    }
#endif
  }

#if ENABLE_GPU
  if (maxGPUTensorSize > 0) {
    cudaError_t cerror ;
    cerror = cudaMalloc(&gpuDispatchMemory, maxGPUTensorSize) ;
    if (cerror != cudaSuccess) {
      LOGERROR
      << "could not allocate GPU memory for dispatch because '"
      << cudaGetErrorString(cerror) << '\'' ;
      gpuDispatchMemory = NULL ;
      return vl::VLE_Cuda ;
    }

#if 1
    cerror = cudaStreamCreateWithFlags(&gpuHelperStream, cudaStreamNonBlocking) ;
    if (cerror != cudaSuccess) {
      LOGERROR
      << "could not create a CUDA stream because '"
      << cudaGetErrorString(cerror) << '\'' ;
      return vl::VLE_Cuda ;
    }
    gpuHelperStreamInitialized = true ;
#endif

    // cerror = cudaEventCreateWithFlags(&gpuHelperEvent, cudaEventDisableTiming) ;
    // if (cerror != cudaSuccess) {
    //   LOGERROR
    //     << "could not create a CUDA event because '"
    //     << cudaGetErrorString(cerror) << '\'' ;
    //   return vl::VLE_Cuda ;
    // }
    // gpuHelperEventInitialized = true ;
  }

  {
    // Pin shared memory (all of it in one go).
    cudaError_t cerror = cudaHostRegister(memoryMap,
                                          memoryMapSize,
                                          cudaHostRegisterDefault) ;
    if (cerror != cudaSuccess) {
      LOGERROR
        << "could not pin memory because of CUDA error '"
        << cudaGetErrorString(cerror) << '\'' ;
    } else {
      LOG(2) << "pinned shared memory" ;
    }
  }
#endif

  return vl::VLE_Success ;
}

// attachPeer
vl::ErrorCode
SharedTensorSpace::attachPeer(int lab)
{
  if (peerTensors.size() != tensors.size()) {
    peerTensors.resize(tensors.size()) ;
  }
  for (int t = 0 ; t < tensors.size() ; ++t) {
    SharedTensorPeerInstance peerTensor ;
    peerTensor.lab = lab ;
    peerTensor.state = SharedTensorSpace::ready ;
    peerTensor.mappedCpuMemory = (char*)memoryMap
    + tensors[t].memoryMapOffset
    + lab * memoryMapLabStride ;
    peerTensor.accumulated = false ;
    peerTensors[t].push_back(peerTensor) ;
  }
  return vl::VLE_Success ;
}

// Destroy all resources
// 1) unmap and unlink shared memory map
// 2) ...

void SharedTensorSpace::finalize()
{
  int error ;

  initialized = false ;

#if ENABLE_GPU
  if (memoryMap) {
    cudaHostUnregister(memoryMap) ;
  }

  // if (gpuHelperEventInitialized) {
  //   cudaEventDestroy(gpuHelperEvent) ;
  //   gpuHelperEventInitialized = false ;
  // }

  if (gpuHelperStreamInitialized) {
    cudaStreamDestroy(gpuHelperStream) ;
    gpuHelperStream = 0 ;
    gpuHelperStreamInitialized = false ;
  }

  if (gpuDispatchMemory) {
    cudaFree(gpuDispatchMemory) ;
    gpuDispatchMemory = NULL ;
  }

  for (tensors_t::iterator T = tensors.begin() ;
       T != tensors.end() ;
       T++)
  {
    if (T->gpuMemory && T->gpuMemoryIsOwned) {
      cudaFree(T->gpuMemory) ;
      T->gpuMemory = NULL ;
      T->gpuMemoryIsOwned = false ;
    }

    if (T->gpuEventIsInitialized) {
      cudaEventDestroy(T->gpuEvent) ;
      T->gpuEvent = 0 ;
      T->gpuEventIsInitialized = false ;
    }
  }
  gpuDevice = -1 ;
#endif

  if (memoryMap) {
    munmap(memoryMap, memoryMapSize) ;
    memoryMap = NULL ;
  }

  error = shm_unlink(memoryMapName.c_str()) ;
  if (error == -1 && errno == EACCES) {
    LOGERROR << "Cannot clear the shared memory map due to a permission error." ;
  }
  memoryMapFD = -1 ;
  tensors.clear() ;
  numLabs = -1 ;
}

void SharedTensorSpace::mexPrint() const
{
  mexPrintf("\tlab %d of %d\n", lab, numLabs) ;
  mexPrintf("\tshared memory: '%s', %d bytes mapped at address: 0x%zx\n",
            memoryMapName.c_str(),memoryMapSize,memoryMap) ;
  for (int tensorIndex = 0 ; tensorIndex < tensors.size() ; ++tensorIndex) {
    SharedTensorInstance const & T = tensors[tensorIndex] ;
    mexPrintf("\tTensor '%s'\n", T.name.c_str()) ;
    mexPrintf("\t\t[") ;
    for (int k = 0 ; k < T.descriptor.shape.getNumDimensions() ; ++k) {
      mexPrintf(" %d", T.descriptor.shape.getDimensions()[k]) ;
    }
    mexPrintf("] %s %s\n",
              T.descriptor.dataType == vl::VLDT_Double?"double":"single",
              T.descriptor.deviceType == vl::VLDT_CPU?"CPU":"GPU") ;
    mexPrintf("\t\tCPU address: 0x%zx\n", T.cpuMemory) ;
    mexPrintf("\t\tGPU address: 0x%zx\n", T.gpuMemory) ;

    if (peerTensors.size() > tensorIndex) {
      for (int p = 0 ; p < peerTensors[tensorIndex].size() ; ++p) {
        SharedTensorPeerInstance const & PT = peerTensors[tensorIndex][p] ;
        mexPrintf("\t\tPeer instance %d\n", p) ;
        mexPrintf("\t\t\tlab: %0d\n", PT.lab) ;
        mexPrintf("\t\t\tmapped CPU address: 0x%zx\n",PT.mappedCpuMemory) ;
      }
    }
  }
}

/* ---------------------------------------------------------------- */
/*                                                      ProcessPool */
/* ---------------------------------------------------------------- */

#pragma mark -

/// Represents a pool of collaborating MATLAB processes. Usually each
/// process corresponds to a certain MATLAB instance in a MATLAB pool.
class ProcessPool
{
public:
  /// Create an un-intialized ProcessPool. Before it is used,
  /// the pool must be initialized using init(). This design allows
  /// to catch errors during initialization without resorting to exceptions.
  ProcessPool() ;

  /// Automatically calls ::finalize().
  ~ProcessPool() ;

  /// Initialize the instance \a lab of \a numLabs pools. The function
  /// timesout.
  vl::ErrorCode init(int lab, int numLabs, SharedTensorSpace * space) ;

  /// Gracefully shutdown the connection with the other processes,
  /// waiting for them to finish updating as needed. After this, the
  /// supervisory thread quits, but the object remains initialized
  /// to allow reading off the final value of the tensor.
  ///
  /// The function timesout.
  vl::ErrorCode shutdown() ;

  /// Immediately terminate the ProcessPool instance and release all
  /// resources.
  void finalize() ;

  /// Print information.
  ///
  /// This function must be called from the MATLAB thread.
  void mexPrint() const ;

  /// Push a tensor in the pool for accumulation.
  ///
  /// This function must be called from the MATLAB thread. It throws
  /// a MEX error on error and can time out.
  void mexPush(std::string const & name, mxArray const * x,
               bool inplace = false) ;

  /// Pull an accumulated tensor from the pool.
  ///
  /// This function must be called from the MATLAB thread. It throws
  /// a MEX error on error and an time out.
  mxArray * mexPull(std::string const & name, bool inplace = false) ;

private:
  bool initialized ;
  uint32_t session ;
  int lab ;
  int numLabs ;
  size_t timeoutInterval ;
  int socketFD ;
  SharedTensorSpace * sharedSpace ;

  // Peer processes.
  struct Peer
  {
    int lab ;
    int socketFD ;
    bool cudaCanAccessPeer ; //cudaDeviceCanAccessPeer
    bool canShutdown ;

    Peer(int lab)
      : lab(lab), socketFD(-1), cudaCanAccessPeer(false), canShutdown(false)
    { }

    bool operator== (int lab) { return this->lab == lab ; }
  } ;

  typedef std::vector<Peer> peers_t ;
  peers_t peers ;

  // Messages between peer processes.
  struct Message
  {
    enum MessageType {
      /// Sent from root to leaves to request initialization during
      /// hanshake.
      init,

      /// Sent from leaves to root to acknowledge initialization.
      initDone,

      /// Sent from root to leaves to request attching the shared
      /// resources (shared memory).
      attach,

      /// Sent to advertise a state change for a tensor.
      tensorStateChange,

      /// Shutdown sequence
      requestShutdown,
      readyToShutdown
    }
    type ;

    // The transaction number.
    size_t transaction ;

    // Sender and destination process indexes.
    int16_t from ;
    int16_t to ;

    // Session identifier, used for sanity checks.
    uint32_t session ;

    // Tensort ID and state for a tensor state change.
    uint32_t tensorId ;
    SharedTensorSpace::SharedTensorState tensorState ;
    Message() : transaction(0), tensorId(0) { }
  } ;

  // Create a socket and wait for children to connect
  vl::ErrorCode makeSocket() ;

  // Connect to parent
  vl::ErrorCode connectParent() ;

  // Delete all sockets, closing the connections with peer processes
  void deleteSockets() ;

  // Send a message to a process, based on its index. The function
  // times out.
  vl::ErrorCode send(Message &msg, int to) ;

  // Receive a message from a process, based on its index.
  //
  // The function times out. If \a timeout is set to zero,
  // the function returns immediately
  // with code vl::VLE_NoData. If \a timeout is negative, the
  // default timeout is used. Otherwise, the specified timeout is used.
  vl::ErrorCode receive(Message &msg, int from, int timeout = -1) ;

  // Supervisory thread.
  tthread::thread * thread ;
  tthread::mutex mutex ;
  tthread::condition_variable condition ;
  int threadError ;
  int threadPipe [2] ;
  bool threadShouldShutdown ;
  enum ThreadState {
    threadPerformingHandshake,
    threadRunning,
    threadShutdownRequested,
    threadQuittingOnError,
    threadDone } threadState ;
  static void threadEntryPoint(void * thing) ;
  void threadLoop() ;
  void threadHandshake() ;
  void threadQuit() ;
} ;


ProcessPool::ProcessPool()
: initialized(false),
lab(-1), numLabs(0),
session(0), thread(NULL),
socketFD(-1)
{
  threadPipe[0] = -1 ;
  threadPipe[1] = -1 ;
}

ProcessPool::~ProcessPool()
{ finalize() ; }

vl::ErrorCode ProcessPool::init(int newLab, int newNumLabs, SharedTensorSpace * newSharedSpace)
{
  vl::ErrorCode error ;
  int parent ;

  assert(newLab >= 0) ;
  assert(newNumLabs > newLab) ;
  assert(newSharedSpace) ;

  // finalize process pool if previously initialized
  finalize() ;

  // set members
  lab = newLab ;
  numLabs = newNumLabs ;
  sharedSpace = newSharedSpace ;
  timeoutInterval = 60UL * 60UL * 1000UL * 1000UL ; // 60s
  //timeoutInterval = 1000000000 ; // 1000s

  // infer parent and children labs
  int bit = ffs(lab) - 1 ;
  if (bit == -1) { bit = 31 ; }

  parent = lab & (~(1 << bit)) ;
  if (parent != lab) {
    // peers[0] always contain the parent (except for root)
    peers.push_back(Peer(parent)) ;
  }

  for (int k = 0 ; k < bit ; ++k) {
    int child = lab | (1 << k) ;
    if (child < numLabs) {
      // Which peers[] gets which children is determined later
      // during hadshake based on the random connection order.
      // Here we assign a provisional lab index using negative indexes
      // as these are needed to use send().
      peers.push_back(Peer(-child)) ;
    }
  }

  error = makeSocket() ;
  if (error != vl::VLE_Success) goto done ;

  error = connectParent() ;
  if (error != vl::VLE_Success) goto done ;

  // create a thread
  threadShouldShutdown = false ;
  threadState = threadPerformingHandshake ;
  thread = new tthread::thread(threadEntryPoint, this) ;

  // wait for handshake to be complete
  {
    tthread::lock_guard<tthread::mutex> lock(mutex) ;
    while (threadState == threadPerformingHandshake) {
      condition.wait(mutex) ;
    }
    if (threadState == threadRunning) {
      error = vl::VLE_Success ;
    } else {
      error = vl::VLE_Unknown ;
    }
  }

done:
  if (error != vl::VLE_Success) {
    finalize() ;
    return error ;
  } else {
    initialized = true ;
    return vl::VLE_Success ;
  }
}

vl::ErrorCode ProcessPool::shutdown()
{
  size_t start = vl::getTime() ;
  threadShouldShutdown = true ;
  // Signal the supervisory thread
  char dummy = 1 ;
  write(threadPipe[1], &dummy, 1) ;

  {
    tthread::lock_guard<tthread::mutex> lock(mutex) ;
    while (threadState == threadRunning ||
           threadState == threadShutdownRequested) {
      if (vl::getTime() > start + timeoutInterval) {
        LOGERROR << "timeout while shutting down" ;
        return vl::VLE_Timeout ;
      }
      condition.wait(mutex) ;
    }
  }

  return vl::VLE_Success ;
}

void ProcessPool::finalize()
{
  if (thread) {
    // Tell thread to quit.
    {
      tthread::lock_guard<tthread::mutex> lock(mutex) ;
      if (threadState != threadQuittingOnError) {
        threadState = threadDone ;
        condition.notify_all() ;
      }
    }
    // Wait for the thread to quit.
    if (thread->joinable()) {
      thread->join() ;
    }
    // Delete the thread object.
    delete thread ;
    thread = NULL ;
  }
  deleteSockets() ;
  peers.clear() ;

  if (sharedSpace) {
    sharedSpace->finalize() ;
    delete sharedSpace ;
    sharedSpace = NULL ;
  }
  lab = -1 ;
  numLabs = 0 ;
  session = 0 ;
  initialized = false ;
}

// make the pipe to or from a peer
vl::ErrorCode ProcessPool::makeSocket()
{
  int error ;
  char socketName [256] ;
  struct sockaddr_un socketAddress ;
  size_t start = vl::getTime() ;
  snprintf(socketName, sizeof(socketName), "/tmp/mcn-socket-%02d", lab) ;

  // Cerate a pipe FD for notification between MATLAB's thread
  // and the supervisory thread. This is needed to allow awaking
  // the supervisory thread.
  error = pipe(threadPipe) ;
  if (error == -1) {
    LOGERROR
    << "cannot create inter-threads pipe because: '"
    << strerror(errno) << '\'' ;
    return vl::VLE_Unknown ;
  }

  // create socket FD
  socketFD = socket(AF_UNIX, SOCK_STREAM, 0) ;
  if (socketFD == -1) {
    LOGERROR
    << "cannot create socket " << socketName
    << "because: " << strerror(errno) ;
    return vl::VLE_Unknown ;
  }

  // copy socket path into socketAddress
  memset(&socketAddress, 0, sizeof(socketAddress)) ;
  socketAddress.sun_family = AF_UNIX;
  strncpy(socketAddress.sun_path, socketName,
          sizeof(socketAddress.sun_path) - 1) ;

  // delete socket path if it exists before binding
  if (access(socketAddress.sun_path, F_OK) == 0) {
    unlink(socketAddress.sun_path) ;
  }

  // bind socket
  error = bind(socketFD,
               (struct sockaddr *)&socketAddress,
               sizeof(socketAddress)) ;

  if (error == -1) {
    LOGERROR
    << "cannot bind socket " << socketName
    << "because: " << strerror(errno) ;
    return vl::VLE_Unknown ;
  }

  // start listening for children connections
  size_t numChildren = peers.size() - (lab > 0) ;
  if (numChildren == 0) return vl::VLE_Success ;

  error = listen(socketFD, numChildren) ;
  if (error == -1) {
    LOGERROR
    << "cannot listen to socket " << socketName
    << "because: " << strerror(errno) ;
    return vl::VLE_Unknown ;
  }

  // do not block on accept()
  fcntl(socketFD, F_SETFL, fcntl(socketFD, F_GETFL, 0) | O_NONBLOCK);

  // Accept one connection per child.
  for (int p = (lab > 0) ; p < peers.size() ; ++p) {
    for (;;) {
      peers[p].socketFD = accept(socketFD, NULL, NULL) ;
      if (peers[p].socketFD == -1) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
          if (vl::getTime() < start + timeoutInterval) continue ; // retry
          LOGERROR
          << "timed out while accepting connection from peer " << peers[p].lab ;
          return vl::VLE_Timeout ;
        }
        LOGERROR
        << " cannot accept connection from peer " << peers[p].lab
        << " because: " << strerror(errno) ;
        return vl::VLE_Unknown ;
      }
      break ;
    }
    fcntl(peers[p].socketFD, F_SETFL,
          fcntl(peers[p].socketFD ,F_GETFL, 0) | O_NONBLOCK) ;
  }

  return vl::VLE_Success ;
}

// Make the pipe to or from a peer.
vl::ErrorCode ProcessPool::connectParent()
{
  int error ;
  char socketName [256] ;
  struct sockaddr_un socketAddress ;
  size_t start = vl::getTime() ;

  if (lab == 0) {
    return vl::VLE_Success ;
  }

  snprintf(socketName, sizeof(socketName),
           "/tmp/mcn-socket-%02d", peers[0].lab) ;

  for (;;) {
    peers[0].socketFD = socket(AF_UNIX, SOCK_STREAM, 0) ;
    if (peers[0].socketFD == -1) {
      if (vl::getTime() < start + timeoutInterval) {
        usleep(30000) ;
        continue ; // try again
      }
      LOGERROR
      << "cannot create socket '" << socketName
      << "' because '" << strerror(errno) << '"' ;
      return vl::VLE_Unknown ;
    }
    break ;
  }
  fcntl(peers[0].socketFD, F_SETFL,
        fcntl(peers[0].socketFD ,F_GETFL, 0) | O_NONBLOCK) ;

  // copy socket path into socketAddress
  memset(&socketAddress, 0, sizeof(socketAddress)) ;
  socketAddress.sun_family = AF_UNIX;
  strncpy(socketAddress.sun_path, socketName,
          sizeof(socketAddress.sun_path) - 1) ;

  // establish connection
  for (int trials = 0 ; ; ++trials) {
    error = connect(peers[0].socketFD,
                    (struct sockaddr *)&socketAddress,
                    sizeof(socketAddress)) ;
    if (error == 0) break ;
    if (vl::getTime() < start + timeoutInterval) {
      usleep(30000) ;
      continue ;
    }
    LOGERROR
    << "cannot connect socket " << socketName
    << " after trying " << trials
    << " times because '" << strerror(errno) << '"' ;
    return vl::VLE_Unknown ;
  }

  return vl::VLE_Success ;
}

void ProcessPool::deleteSockets()
{
  char socketName [256] ;
  snprintf(socketName, sizeof(socketName), "/tmp/mcn-socket-%02d", lab) ;

  for (int p = 0 ; p < peers.size() ; ++p) {
    if (peers[p].socketFD != -1) {
      close(peers[p].socketFD) ;
      peers[p].socketFD = -1 ;
    }
  }
  if (socketFD != -1) {
    close(socketFD) ;
    socketFD = -1 ;
  }
  for (int t = 1 ; t >= 0 ; --t) {
    if (threadPipe[t] != -1) {
      close(threadPipe[t]) ;
      threadPipe[t] = -1 ;
    }
  }
  unlink(socketName) ;
}

vl::ErrorCode ProcessPool::send(Message & msg, int to)
{
  // Find connection to peer.
  peers_t::const_iterator rel = std::find(peers.begin(), peers.end(), to) ;
  assert(rel != peers.end()) ;

  // Add complementery information to the message.
  msg.session = session ;
  msg.from = lab ;
  msg.to = to ;

  // Send all bytes.
  {
    int bytesWritten = 0 ;
    int status ;
    char * nextByte = (char*)&msg ;
    while (bytesWritten < sizeof(msg)) {
      status = write(rel->socketFD, nextByte, sizeof(msg) - bytesWritten) ;
      if (status == -1) {
        LOGERROR
        << "could not send message to " << to
        << " because '" << strerror(errno) << '\'' ;
        return vl::VLE_Unknown ;
      }
      bytesWritten += status ;
    }
  }

  LOG(3)
  << "sent message to " << to
  << " (type "  << msg.type
  << ", state " << msg.tensorState
  << " tensor " << msg.tensorId
  << ')' ;

  return vl::VLE_Success ;
}

vl::ErrorCode ProcessPool::receive(Message & msg, int from, int timeout)
{
  size_t waited = 0 ; // microsecond
  size_t const pollInterval = 100 ;

  if (timeout < 0) { timeout = timeoutInterval ; }

  // find connection to peer
  peers_t::const_iterator rel = std::find(peers.begin(), peers.end(), from) ;
  assert(rel != peers.end()) ;

  // receive all bytes
  {
    int bytesRead = 0 ;
    int status ;
    char * nextByte = (char*)&msg ;
    while (bytesRead < sizeof(msg)) {
      status = read(rel->socketFD, nextByte, sizeof(msg) - bytesRead) ;
      if (status == 0 || status == -1) {
        if (status == 0 || errno == EAGAIN) {
          if (timeout == 0 && bytesRead == 0) {
            // non blocking operation, no message, just return no data
            return vl::VLE_NoData ;
          }
          if (timeout > 0 && waited >= timeout) {
            if (verbosity >= 1) {
              LOGERROR
              << "timed out while receiving a message from lab " << from
              << " because '" << strerror(errno) << '\'' ;
            }
            return vl::VLE_Timeout ;
          }
          usleep(pollInterval) ;
          waited += pollInterval ;
          continue ;
        }
        if (verbosity >= 1) {
          LOGERROR
          << "error while receiving a message from lab " << from
          << ": '" << strerror(errno) << '\'' ;
        }
        return vl::VLE_Unknown ;
      }
      bytesRead += status ;
    }
  }

  // check message integrity
  if ((msg.type != Message::init &&
       msg.type != Message::initDone)
      && (msg.session != session &&
          msg.from != from &&
          msg.to != lab)) {
        LOGERROR
        << "received an unexpected message from lab " << from
        << "\n\tmsg: session:" << msg.session
        << " from:" << msg.from
        << " to:"  << msg.to
        << " type:" << msg.type
        << "\n\tthis session:" << session ;
        return vl::VLE_Unknown ;
      }

  LOG(3)
  << "received message from "<<from
  << " (type "   << msg.type
  << ", state "  << msg.tensorState
  << ", tensor " << msg.tensorId
  << ')' ;

  return vl::VLE_Success ;
}

void ProcessPool::mexPrint() const
{
  tthread::lock_guard<tthread::mutex> (mutex) ;
  sharedSpace->mexPrint() ;
}

// Push this data

void ProcessPool::mexPush(std::string const & name, mxArray const * x, bool inplace)
{
  tthread::lock_guard<tthread::mutex> lock(mutex) ;

  // Make sure that the session did not end.
  if (threadState != threadRunning) {
    vlmxError(VLMXE_Execution, "The connection to other MATLAB instances shut down.") ;
  }

  // Search tensor by name.
  SharedTensorSpace::tensors_t::iterator T
  = std::find(sharedSpace->tensors.begin(), sharedSpace->tensors.end(), name) ;
  if (T == sharedSpace->tensors.end()) {
    vlmxError(VLMXE_IllegalArgument, "There is no tensor '%s'.", name.c_str()) ;
  }

  // Encapsulate MATLAB argument and check tensor compatibility.
  vl::MexTensor mtens(context) ;
  mtens.init(x) ;

  if (mtens.getDeviceType() != T->descriptor.deviceType) {
    vlmxError(VLMXE_IllegalArgument, "The tensor device type is incorrect.") ;
  }

  if (mtens.getDataType() != T->descriptor.dataType) {
    vlmxError(VLMXE_IllegalArgument, "The tensor data type is incorrect.") ;
  }

  if (mtens.getNumElements() != T->descriptor.shape.getNumElements()) {
    vlmxError(VLMXE_IllegalArgument, "The tensor shape is incorrect.") ;
  }

  if (inplace && T->descriptor.deviceType != vl::VLDT_GPU) {
    vlmxError(VLMXE_IllegalArgument, "Inplace operations are supported only for GPU arrays.") ;
  }

  // Wait until ready to push.
  {
    size_t start = vl::getTime() ;
    while (T->state != SharedTensorSpace::ready) {
      if ((vl::getTime() - start) > timeoutInterval) {
        vlmxError(VLMXE_TimeOut, "PUSH operation timed out on tensor '%s'.", T->name.c_str()) ;
      }
      if (threadState != threadRunning) {
        vlmxError(VLMXE_Execution, "The connection to other MATLAB instances shut down.") ;
      }
      condition.wait(mutex) ;
    }
  }

  // Copy memory to SharedSpace
  if (T->descriptor.deviceType == vl::VLDT_CPU) {
    memcpy(T->cpuMemory, mtens.getMemory(), T->descriptor.getSizeInBytes()) ;
  } else {
#if ENABLE_GPU
    cudaError_t cerror ;

    // sync main thread (do not start until the parameters have been computed!)
    cudaEventRecord(T->gpuEvent, 0) ;
    cudaStreamWaitEvent(sharedSpace->gpuHelperStream, T->gpuEvent, 0) ;

    if (inplace) {
      if (T->gpuMemoryIsOwned && T->gpuMemory) {
        // Free the previously allocated memory as we are going to use
        // an inplace operation on this tensor.
        cudaFree(T->gpuMemory) ;
        T->gpuMemory = NULL ;
      }
      T->gpuMemoryIsOwned = false ;
      T->gpuMemory = mtens.getMemory() ;
    } else {
      if (T->gpuMemoryIsOwned == false || T->gpuMemory == NULL) {
        cerror = cudaMalloc(&T->gpuMemory,
                            T->descriptor.getSizeInBytes()) ;
        if (cerror != cudaSuccess) {
          T->gpuMemory = NULL ;
          T->gpuMemoryIsOwned = false ;
          vlmxError(VLMXE_Alloc, "CUDA error while allocating GPU memory (%s).",
                    cudaGetErrorString(cerror)) ;
        }
        T->gpuMemoryIsOwned = true ;
        cerror = cudaMemcpyAsync (T->gpuMemory,
                                  mtens.getMemory(),
                                  T->descriptor.getSizeInBytes(),
                                  cudaMemcpyDeviceToDevice,
                                  sharedSpace->gpuHelperStream) ;
        if (cerror != cudaSuccess) {
          vlmxError(VLMXE_Execution, "CUDA error while copying GPU data (%s).",
                    cudaGetErrorString(cerror)) ;
        }
      }
    }
#endif
  }
  T->transaction ++ ;
  T->numChildrenToAccumulate = 0 ;
  for (int p = (lab > 0) ; p < peers.size() ; ++p) {
    SharedTensorSpace::SharedTensorPeerInstance & PT = sharedSpace->peerTensors[T - sharedSpace->tensors.begin()][p] ;
    PT.accumulated = false ;
    T->numChildrenToAccumulate += 1;
  }
  asm volatile("": : :"memory") ; // Memory barrier: prevents compiler from reordering
  T->state = SharedTensorSpace::accumulateChildren ; // Must be last to close transaction

  // Signal the supervisory thread
  {
    char dummy = 1 ;
    write(threadPipe[1], &dummy, 1) ;
  }
}

mxArray * ProcessPool::mexPull(std::string const & name, bool inplace)
{
  tthread::lock_guard<tthread::mutex> lock(mutex) ;

  // Search the tensor with the specified name.
  SharedTensorSpace::tensors_t::const_iterator T
  = std::find(sharedSpace->tensors.begin(), sharedSpace->tensors.end(), name) ;

  if (T == sharedSpace->tensors.end()) {
    vlmxError(VLMXE_IllegalArgument, "There is no tensor with the specified name.") ;
  }

  if (inplace && T->descriptor.deviceType != vl::VLDT_GPU) {
    vlmxError(VLMXE_IllegalArgument, "Inplace operations are supported only for GPU arrays.") ;
  }

  // Wait until the tensor is in a ready state, or a timeout occurs.
  {
    size_t start = vl::getTime() ;
    while (T->state != SharedTensorSpace::ready) {
      if ((vl::getTime() - start) > timeoutInterval) {
        vlmxError(VLMXE_TimeOut, "PULL operation timed out on tensor %s.", T->name.c_str()) ;
      }
      if (threadState != threadRunning &&
          threadState != threadShutdownRequested) {
        // If the connection is down, there is no hope of recovering from this
        // situation; in fact a deadlock can be expected as with the thread down
        // the hearbeat stops and there is no way of timing out the mutex.
        vlmxError(VLMXE_Execution,
                  "The tensor is not ready and the connection to other MATLAB instances was lost.") ;
      }
      condition.wait(mutex) ;
    }
  }

  if (inplace) {
    // With in-place operations, the only purpose of pull() is to wait until
    // the tensor is ready and can be accessed.
    return NULL ;
  } else {
    vl::MexTensor result(context) ;
    result.init(T->descriptor.deviceType, T->descriptor.dataType, T->descriptor.shape) ;

    if (T->descriptor.deviceType == vl::VLDT_CPU) {
      memcpy(result.getMemory(),
             T->cpuMemory,
             T->descriptor.getSizeInBytes()) ;
    } else {
#if ENABLE_GPU
      // Synchronous with main thread.
      cudaError_t cerror = cudaMemcpyAsync (result.getMemory(),
                                           T->gpuMemory,
                                           T->descriptor.getSizeInBytes(),
                                           cudaMemcpyDeviceToDevice,
                                           sharedSpace->gpuHelperStream) ;
      if (cerror != cudaSuccess) {
        vlmxError(VLMXE_Execution, "CUDA error while copying GPU data (%s).",
                  cudaGetErrorString(cerror)) ;
      }

      cerror = cudaStreamSynchronize(sharedSpace->gpuHelperStream) ;
      if (cerror != cudaSuccess) {
        vlmxError(VLMXE_Execution, "CUDA error while synchronizing a stream (%s).",
                  cudaGetErrorString(cerror)) ;
      }
#endif
    }
    return result.relinquish() ;
  }
}

#pragma mark -

void ProcessPool::threadEntryPoint(void * thing)
{
  ((ProcessPool*)thing)->threadLoop() ;
}

// The purpose of the handshake sequence is to make sure that
// all processes are properly communicating and ready to go.
// It is also required to synchornize the root (which creates several
// shared resources) and the other nodes (which attach them).

void ProcessPool::threadHandshake()
{
  Message msg ;
  vl::ErrorCode error = vl::VLE_Success ;

  LOG(2) << "begin" ;

  // make sure the supervisory thread operates on the same CUDA device as the main thread
#if ENABLE_GPU
  {
    if (sharedSpace->gpuDevice >= 0) {
      cudaError_t cerror = cudaSetDevice(sharedSpace->gpuDevice) ;
      if (cerror != cudaSuccess) {
        LOGERROR << "could not switch supervisory thread to CUDA device " << sharedSpace->gpuDevice ;
        error = vl::VLE_Cuda ;
        goto done ;
      }
      LOG(2) << "supervisory thread switched to CUDA device " << sharedSpace->gpuDevice ;
    }
  }
#endif

  // receive message from parent (except for root)
  if (lab == 0) {
    session = (uint32_t)vl::getTime() ;
    // root atteches first
    error = sharedSpace->attach(0, numLabs) ;
    if (error != vl::VLE_Success) {
      LOGERROR << "root could not attache shared space" ;
      error = vl::VLE_Unknown ;
      goto done ;
    }
    LOG(2) << "root attached the shared tensor space" ;
  } else {
    error = receive(msg, peers[0].lab) ;
    if (error != vl::VLE_Success || msg.type != Message::init) {
      LOGERROR << "did not receive a message from parent" ;
      error = vl::VLE_Unknown ;
      goto done ;
    }
    session = msg.session ;
    // children attach now
    error = sharedSpace->attach(lab, numLabs) ;
    if (error != vl::VLE_Success || msg.type != Message::init) {
      LOGERROR << "could not attach shared space" ;
      error = vl::VLE_Unknown ;
      goto done ;
    }
    LOG(2) << "child attached the shared tensor space" ;
  }

  // send message to all children
  for (int p = (lab > 0) ; p < peers.size() ; ++p) {
    msg.type = Message::init ;
    error = send(msg,peers[p].lab) ;
    if (error != vl::VLE_Success) {
      LOGERROR << "could not send a message to a child" ;
      goto done ;
    }
  }

  // receive message from all children
  for (int p = (lab > 0) ; p < peers.size() ; ++p) {
    error = receive(msg,peers[p].lab) ;
    if (error != vl::VLE_Success || msg.type != Message::initDone) {
      error = vl::VLE_Unknown ;
      goto done ;
    }
    // now we can identify the child lab index
    peers[p].lab = msg.from ;
  }

  // register peer tensors in the same order as peer[]
  for (int p = 0 ; p < peers.size() ; ++p) {
    sharedSpace->attachPeer(peers[p].lab) ;
  }

  // send message to parent (excep for root)
  if (lab > 0) {
    msg.type = Message::initDone ;
    error = send(msg, peers[0].lab) ;
    if (error != vl::VLE_Success) {
      error = vl::VLE_Unknown ;
      goto done ;
    }
    session = msg.session ;
  }

done:
  if (error != vl::VLE_Success) {
    // The handshake was unsuccesful. The supervisory thread
    // will quit.
    threadState = threadQuittingOnError ;
    LOGERROR << "handshake failed" ;
  } else {
    // The handshake was successful. The supervisory thread can
    // start running.
    threadState = threadRunning ;
    LOG(2) << "handshake successful" ;
  }
  condition.notify_all() ;
}

// On quitting, the supervisory thread closes the communication
// sockets, which causes the superisory threads in other processes
// to quit as well. However, this does *not* reinitialize the process
// pool structure so that it is still possible to read the last
// value written in the tensors.

void ProcessPool::threadQuit()
{
  // Shut down connections
  LOG(2) << "supervisor thread quitting" ;
  deleteSockets() ;
  condition.notify_all() ;
}

void ProcessPool::threadLoop()
{
  threadHandshake() ;

  int pollStatus = 0 ;
  bool notificationPending = false ;
  size_t const pollInterval = 499UL * 1000UL; // allow heartbeats (mus)
  size_t const heartbeatInterval = 500UL * 1000UL * 1000UL ; // (ns)
  size_t lastHeartbeat = vl::getTime() ;

  struct pollfd * polls = new struct pollfd [peers.size() + 1] ;
  for (int p = 0 ; p < peers.size() ; ++p) {
    polls[p].fd = peers[p].socketFD ;
    polls[p].events = POLLIN | POLLHUP | POLLERR | POLLNVAL ;
  }
  polls[peers.size()].fd = threadPipe[0] ;
  polls[peers.size()].events = POLLIN ;

  for (; //tthread::lock_guard<tthread::mutex> lock(mutex) ;
       threadState != threadQuittingOnError &&
         threadState != threadDone ;)
  {
    size_t now = vl::getTime() ;

    // Regular heartbeats are used to occassionally wake up a waiting
    // main thread and allow it to time out on pull(), push(),
    // and other operations.
    notificationPending |= (now > lastHeartbeat + heartbeatInterval) ;
    if (notificationPending) {
      condition.notify_all() ;
      lastHeartbeat = now ;
    }

    // Wait for incoming messages or a timeout.
    pollStatus = poll(polls, peers.size() + 1, pollInterval) ;
    if (pollStatus < 0) {
      threadState = threadQuittingOnError ;
      continue ;
    }

    // Check for threadPipe events sent from the main thread.
    if (polls[peers.size()].revents & POLLIN) {
      LOG(3) << "supervisory thread notified from main thread" ;
      char dummy ;
      read(threadPipe[0], &dummy, 1) ;
    }

    for (int p = 0 ; p < peers.size()
         && threadState != threadQuittingOnError ;
         ++ p)
    {
      // Check for communication errors.
      if (polls[p].revents & (POLLHUP | POLLERR | POLLNVAL)) {
        threadState = threadQuittingOnError ;
        continue ;
      }

      // Skip this peer if there is no incoming data.
      if ((polls[p].revents & POLLIN) == 0) continue ;

      // Receive the message.
      Message msg ;
      if (receive(msg, peers[p].lab) != vl::VLE_Success) {
        LOGERROR << "connection with " << peers[p].lab << " was lost." ;
        threadState = threadQuittingOnError ;
        continue ;
      }

      // Process the message.
      switch (msg.type) {
        case Message::tensorStateChange:
        {
          LOG(3)
          << "received tensor state change from lab " << msg.from
          << " for tensor " << sharedSpace->tensors[msg.tensorId].name.c_str()
          << " to state " << msg.tensorState
          << " for transaction " << msg.transaction ;
          SharedTensorSpace::SharedTensorPeerInstance & T
          = sharedSpace->getPeerTensor(msg.tensorId, msg.from) ;
          T.state = msg.tensorState ;
          T.transaction = msg.transaction ;
          break ;
        }

        case Message::requestShutdown:
        {
          threadState = threadShutdownRequested ;
          LOG(3) << "received shutdown request, propagating to other labs" ;
          // Propagate to all peers minus the source of this message.
          int sourcePeer = msg.from ;
          for (int q = 0 ; q < peers.size() ; ++q) {
            if (sourcePeer == peers[q].lab) continue ;
            vl::ErrorCode error = send(msg, peers[q].lab) ;
            if (error != vl::VLE_Success) {
              threadState = threadQuittingOnError ;
              break ;
            }
          }
          break ;
        }

        case Message::readyToShutdown:
        {
          peers_t::iterator P = std::find(peers.begin(), peers.end(), msg.from) ;
          P->canShutdown = true ;
          LOG(3) << "child " << P->lab << " is ready to shutdown" ;
          break ;
        }

        default:
          // Unexpected message.
          LOGERROR << "received an unexpected message, quitting." ;
          threadState = threadQuittingOnError ;
          break ;
      }
    }

    // A flag to check whether all tensors are in a ready state.
    bool allTensorsAreReady = true ;

    // Check all tensors for actions.
    for (int tensorIndex = 0 ; tensorIndex < sharedSpace->tensors.size()
         && threadState != threadQuittingOnError ; ++tensorIndex)
    {
      vl::ErrorCode error = vl::VLE_Success ;
      SharedTensorSpace::SharedTensorInstance & T = sharedSpace->tensors[tensorIndex] ;
      allTensorsAreReady &= (T.state == SharedTensorSpace::ready) ;

      SharedTensorSpace::SharedTensorState currentState ;
      do {
        currentState = T.state ;
        LOG(3) << "visiting tensor " << T.name << " in state " << T.state ;
        switch (T.state)
        {
          case SharedTensorSpace::ready:
            // Nothing to do if ready.
            break ;

          case SharedTensorSpace::accumulateChildren :
          {
            // Search for children that can and should
            // be accumulated for this transaction.
            for (int p = (lab > 0) ; p < peers.size() ; ++p) {
              int peerLab = peers[p].lab ;
              SharedTensorSpace::SharedTensorPeerInstance & PT = sharedSpace->getPeerTensor(tensorIndex, peerLab) ;
              if (PT.transaction == T.transaction &&
                  PT.state == SharedTensorSpace::waitParent &&
                  PT.accumulated == false) {
                switch (T.descriptor.deviceType) {

                  case vl::VLDT_CPU: {
                    switch (T.descriptor.dataType) {
                      case vl::VLDT_Float:
                        vl::impl::blas<vl::VLDT_CPU,vl::VLDT_Float>::axpy
                        (context,
                         T.descriptor.shape.getNumElements(),
                         1.0f,
                         (float*)PT.mappedCpuMemory, 1,
                         (float*)T.cpuMemory, 1) ;
                        break ;

                      case vl::VLDT_Double:
                        vl::impl::blas<vl::VLDT_CPU,vl::VLDT_Double>::axpy
                        (context,
                         T.descriptor.shape.getNumElements(),
                         1.0,
                         (double*)PT.mappedCpuMemory, 1,
                         (double*)T.cpuMemory, 1) ;
                        break ;

                      default:
                        assert(false) ;
                        break ;
                    }
                    break ;
                  }

                  case vl::VLDT_GPU: {
#if ENABLE_GPU
                    if (T.gpuMemory == NULL) {
                      LOGERROR << "internal error: GPU memory not allocated for tensor " << T.name ;
                      error = vl::VLE_Unknown ;
                      break ;
                    }

                    cudaError_t cerror
                    = cudaMemcpyAsync(sharedSpace->gpuDispatchMemory,
                                      PT.mappedCpuMemory,
                                      T.descriptor.getSizeInBytes(),
                                      cudaMemcpyHostToDevice,
                                      sharedSpace->gpuHelperStream) ;
                    if (cerror != cudaSuccess) {
                      LOGERROR
                      << "CUDA error while copying data from host to device ("
                      << cudaGetErrorString(cerror) << ')' ;
                      error = vl::VLE_Cuda ;
                      break ;
                    }

                    cudaStream_t previousStream = context.getCudaHelper().getStream() ;
                    error = context.getCudaHelper().setStream(sharedSpace->gpuHelperStream) ;
                    if (error != vl::VLE_Success) {
                      LOGERROR << "switching CUDA streams:" << context.getLastErrorMessage() ;
                      break ;
                    }

                    switch (T.descriptor.dataType) {
                      case vl::VLDT_Float:
                        error = vl::impl::blas<vl::VLDT_GPU,vl::VLDT_Float>::axpy
                        (context,
                         T.descriptor.shape.getNumElements(),
                         1.0f,
                         (float*)sharedSpace->gpuDispatchMemory, 1,
                         (float*)T.gpuMemory, 1) ;
                        break ;

                      case vl::VLDT_Double:
                        error = vl::impl::blas<vl::VLDT_GPU,vl::VLDT_Double>::axpy
                        (context,
                         T.descriptor.shape.getNumElements(),
                         1.0,
                         (double*)sharedSpace->gpuDispatchMemory, 1,
                         (double*)T.gpuMemory, 1) ;
                        break ;

                      default:
                        assert(false) ;
                        break ;
                    }

                    context.getCudaHelper().setStream(previousStream) ;

                    if (error != vl::VLE_Success) {
                      LOGERROR << "summing tensors:" << context.getLastErrorMessage() ;
                      break ;
                    }
#endif
                    break ;
                  }

                  default:
                    assert(false) ;
                    break ;
                }
                PT.accumulated = true ;
                -- T.numChildrenToAccumulate ;
                LOG(3) << "accumulated child " << PT.lab << "; " << T.numChildrenToAccumulate << " remaining" ;
              }
            }
            // If all children have been accumulated, then
            // notify parent and switch to waitParent state.
            // Note that we change the PT state too as the peer
            // will switch to that upon receiving the notification.
            //
            // The root is a special case because it
            // does not have a parent, so it can switch
            // directly to the waitChildren state. However, in order
            // to reuse the generic code above, we also set it
            // to waitParent and let the next iteration pick this up.
            if (T.numChildrenToAccumulate == 0) {
#if ENABLE_GPU
              if (T.descriptor.deviceType == vl::VLDT_GPU) {
                cudaError_t cerror ;
                if (T.gpuMemory == NULL) {
                  LOGERROR
                  << "internal: GPU memory not allocated for tensor "
                  << T.name ;
                  error = vl::VLE_Unknown ;
                  break ;
                }

                cerror = cudaMemcpyAsync(T.cpuMemory,
                                         T.gpuMemory,
                                         T.descriptor.getSizeInBytes(),
                                         cudaMemcpyDeviceToHost,
                                         sharedSpace->gpuHelperStream) ;
                if (cerror != cudaSuccess) {
                  LOGERROR
                  << "CUDA error while copying from device to host ("
                  << cudaGetErrorString(cerror) << ")" ;
                  error = vl::VLE_Cuda ;
                  break ;
                }

                // TODO: with more granularity, it would be possible for the thread
                // to move onto something else while we wait for the copy to finish.
                // However, this is a little difficult to do with the poll() waiting above.

                cerror = cudaStreamSynchronize(sharedSpace->gpuHelperStream) ;
                if (cerror != cudaSuccess) {
                  LOGERROR
                    << "CUDA error while synchronizing a stream: '"
                    << cudaGetErrorString(cerror) << '\'' ;
                  error = vl::VLE_Cuda ;
                  break ;
                }
              }
              if (error != vl::VLE_Success) break ;
#endif
              T.state = SharedTensorSpace::waitParent ;
              if (lab > 0) {
                int parentLab = peers[0].lab ;
                sharedSpace->getPeerTensor(tensorIndex, parentLab).state = SharedTensorSpace::waitParent ;
                Message msg ;
                msg.type = Message::tensorStateChange ;
                msg.tensorId = tensorIndex ;
                msg.tensorState = T.state ;
                msg.transaction = T.transaction ;
                error = send(msg, parentLab) ;
              }
            }
            break ;
          }

          case SharedTensorSpace::waitParent :
          {
            if (lab > 0) {
              // Check if parent finished updating. If so, we can copy its value here
              // and notify the children to copy us by switching to waitParent state and
              // notifying the children. Note that we change the children peer state too
              // as these peers will switch to that upon being notified.
              int parentLab = peers[0].lab ;
              SharedTensorSpace::SharedTensorPeerInstance & PT
              = sharedSpace->getPeerTensor(tensorIndex, parentLab) ;

              bool parentDone = (PT.transaction == T.transaction &&
                                 PT.state == SharedTensorSpace::waitChildren) ;
              if (!parentDone) continue ;
              switch (T.descriptor.deviceType) {
                case vl::VLDT_CPU:
                  memcpy(T.cpuMemory, PT.mappedCpuMemory, T.descriptor.getSizeInBytes()) ;
                  break ;

                case vl::VLDT_GPU: {
#if ENABLE_GPU
                  if (T.gpuMemory == NULL) {
                    LOGERROR << "internal: GPU memory not allocated for tensor " << T.name ;
                    error = vl::VLE_Unknown ;
                    break ;
                  }
                  cudaError_t cerror = cudaMemcpyAsync(T.gpuMemory,
                                                       PT.mappedCpuMemory,
                                                       T.descriptor.getSizeInBytes(),
                                                       cudaMemcpyHostToDevice,
                                                       sharedSpace->gpuHelperStream) ;
                  if (cerror != cudaSuccess) {
                    LOGERROR
                      << "propagating parent to children: CUDA error while copying from host to device: '"
                      << cudaGetErrorString(cerror) << '\'' ;
                    error = vl::VLE_Cuda ;
                  }
#endif
                  break ;
                }
              }
              if (error != vl::VLE_Success) break ;
            }

            // We have copied data from parent (or there is no parent at all)
            // so we are ready to pass our data to the children and to release
            // the parent from waiting on us.
#if ENABLE_GPU
            if (T.descriptor.deviceType == vl::VLDT_GPU
                && peers.size() > (lab > 0) // There are children
                ) {

              cudaError_t cerror
              = cudaMemcpyAsync(T.cpuMemory,
                                T.gpuMemory,
                                T.descriptor.getSizeInBytes(),
                                cudaMemcpyDeviceToHost,
                                sharedSpace->gpuHelperStream) ;
              if (cerror != cudaSuccess) {
                LOGERROR
                << "propagating children to parent: CUDA error while copying from device to host ("
                << cudaGetErrorString(cerror) << ')' ;
                error = vl::VLE_Cuda ;
              }

              // This is synchrnous, so we can correctly notify
              // that the memory is ready to the peer.
              cerror = cudaStreamSynchronize(sharedSpace->gpuHelperStream) ;
              if (cerror != cudaSuccess) {
                LOGERROR
                  << "CUDA error while synchronizing a stream ("
                  << cudaGetErrorString(cerror) << ")" ;
                error = vl::VLE_Cuda ;
                break ;
              }
            }
            if (error != vl::VLE_Success) break ;
#endif
            T.state = SharedTensorSpace::waitChildren ;
            for (int p = 0 ; p < peers.size() ; ++p) {
              int peerLab = peers[p].lab ;
              SharedTensorSpace::SharedTensorPeerInstance & PT
              = sharedSpace->getPeerTensor(tensorIndex, peerLab) ;
              PT.state = (lab > 0 && p == 0) ? SharedTensorSpace::ready : SharedTensorSpace::waitChildren ;
              Message msg ;
              msg.type = Message::tensorStateChange ;
              msg.transaction = T.transaction ;
              msg.tensorId = tensorIndex ;
              msg.tensorState = (lab > 0 && p == 0) ? SharedTensorSpace::ready : SharedTensorSpace::waitChildren ;
              error = send(msg, peerLab) ;
              if (error != vl::VLE_Success) break ;
            }
            break ;
          }

          case SharedTensorSpace::waitChildren :
          {
            // Check if all children finished updating. If so, we can switch
            // to ready state and notify the parent.
            // Note that we change the peer children state too
            // as these peers will switch to that upon being notified.

            bool allChildrenDone = true ;
            for (int p = (lab > 0) ; p < peers.size() ; ++p) {
              int peerLab = peers[p].lab ;
              SharedTensorSpace::SharedTensorPeerInstance & PT
              = sharedSpace->getPeerTensor(tensorIndex, peerLab) ;
              allChildrenDone &= ((PT.transaction == T.transaction &&
                                   PT.state == SharedTensorSpace::ready) ||
                                  PT.transaction > T.transaction) ;
            }
            if (allChildrenDone) {
              // We already nofified a ready state to the partent before
              asm volatile("": : :"memory") ; // probably overkill here
              T.state = SharedTensorSpace::ready ;
              notificationPending = true ;
            }
            break ;
          }
        }
      } while (T.state != currentState) ;

      if (error != vl::VLE_Success) {
        LOGERROR << "supervisory thread caught an error and will quit." ;
        threadState = threadQuittingOnError ;
      }
    } // check next tensor

    if (threadShouldShutdown && threadState == threadRunning) {
      threadState = threadShutdownRequested;
      LOG(2) << "shutdown sequence initiated" ;
      for (int p = 0 ; p < peers.size() ; ++p) {
        Message msg ;
        msg.type = Message::requestShutdown ;
        send(msg, peers[p].lab) ;
      }
    }

    if (allTensorsAreReady && threadState == threadShutdownRequested) {
      // check if all children are ready
      bool allChildrenCanShutdown = true ;
      for (int p = (lab > 0) ; p < peers.size() ; ++p) {
        allChildrenCanShutdown &= peers[p].canShutdown ;
      }
      if (allChildrenCanShutdown) {
        if (lab == 0) {
          LOG(3) << "the complete tree is ready, shutdown" ;
          threadState = threadDone ;
          continue ;
        }
        // tell parent we are ready
        LOG(2) << "subtree ready to shutdown, telling parent lab" ;
        Message msg ;
        msg.type = Message::readyToShutdown ;
        vl::ErrorCode error = send(msg, peers[0].lab) ;
        if (error != vl::VLE_Success) {
          threadState = threadQuittingOnError ;
          continue ;
        }
      }
    }
  } // go to poll

  delete [] polls ;
  threadQuit() ;
}

/* ---------------------------------------------------------------- */
/*                                                          Context */
/* ---------------------------------------------------------------- */

ProcessPool processPool ;

/*
 Resetting the context here resolves a crash when MATLAB quits and
 the ~Context function is implicitly called on unloading the MEX file.
 */

void atExit()
{
  context.clear() ;
}

/* ---------------------------------------------------------------- */
/*                                                       MEX driver */
/* ---------------------------------------------------------------- */

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;

  enum Commands { init, stats, reset, push, pull } command ;
  bool inplace = false ;
  std::string tensorName ;
  mxArray const * arg ;
  vl::ErrorCode error = vl::VLE_Success ;
  size_t labIndex = 0 ;
  size_t numLabs = 0 ;

  verbosity = 0 ;

  mexAtExit(atExit) ;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  if (nin < 1) {
    mexErrMsgTxt("There are no arguments") ;
  }

  if (!vlmxIsString(in[0], -1)) {
    mexErrMsgTxt("COMMAND is not a string.") ;
  }

  if (vlmxCompareToStringI(in[0],"init") == 0) {
    command = init ;
    if (nin < 4) {
      mexErrMsgTxt("Less than three arguments passed to INIT.") ;
    }
    arg = in[1] ;
    if (!vlmxIsPlainScalar(in[2])) {
      mexErrMsgTxt("LABINDEX is not a plain scalar.") ;
    }
    labIndex = mxGetScalar(in[2]) ;
    if (labIndex < 1) {
      mexErrMsgTxt("LABINDEX must be an integer greater than 0.") ;
    }
    if (!vlmxIsPlainScalar(in[3])) {
      mexErrMsgTxt("NUMLABS is not a plain scalar.") ;
    }
    numLabs = mxGetScalar(in[3]) ;
    if (numLabs < labIndex) {
      mexErrMsgTxt("NUMLABS must be an integer greater or equal to LABINDEX.") ;
    }
    next = 4 ;
  } else if (vlmxCompareToStringI(in[0], "stats") == 0)  {
    command = stats ;
    next = 1 ;
  } else if (vlmxCompareToStringI(in[0], "reset") == 0)  {
    command = reset ;
    next = 1 ;
  } else if (vlmxCompareToStringI(in[0], "push") == 0) {
    if (nin < 3) {
      mexErrMsgTxt("Less than three arguments passed to PUSH.") ;
    }
    command = push ;
    vlmxParseString(tensorName, in[1]) ;
    arg = in[2] ;
    next = 3 ;
  } else if (vlmxCompareToStringI(in[0], "pull") == 0) {
    if (nin < 2) {
      mexErrMsgTxt("Less than two arguments passed to PULL.") ;
    }
    command = pull ;
    vlmxParseString(tensorName, in[1]) ;
    next = 2 ;
  }
  else {
    mexErrMsgTxt("Unknown COMMAND string.") ;
  }

  // optional arguments
  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
      case opt_verbose :
        ++ verbosity ;
        break ;

      case opt_inplace :
        inplace = true ;
        break ;
    }
  }

  switch (command) {
    case init:
    {
      (verbosity >= 2) && mexPrintf("vl_tflowmex: command 'init'\n") ;

      // Initialize shared space. mexInit() may thorow a MEX error;
      // the auto_ptr should avoid a leak in this case.
      std::auto_ptr<SharedTensorSpace> sharedSpace(new SharedTensorSpace()) ;
      sharedSpace->mexInit(arg) ;

      // Initialize the pool, including attaching the shared space.
      // Now the shared space is owned by the process pool.
      error = processPool.init(labIndex - 1, numLabs, sharedSpace.release()) ;
      if (error != vl::VLE_Success) {
        mexErrMsgTxt("Could not initialize connections to other MATLAB labs.") ;
      }

      // At this point, sharedSpace is handled by the ProcessPool thread,
      // so we interact with it indirectly
      break ;
    }

    case stats :
      (verbosity >= 2) && mexPrintf("vl_tflowmex: command 'stats'\n") ;
      processPool.mexPrint() ;
      break ;

    case push :
      (verbosity >= 2) && mexPrintf("vl_tflowmex: command 'push' on tensor '%s'%s\n", tensorName.c_str(), inplace?" (inplace)":"") ;
      processPool.mexPush(tensorName, arg, inplace) ;
      break ;

    case pull :
      (verbosity >= 2) && mexPrintf("vl_tflowmex: command 'pull' on tensor '%s'%s\n", tensorName.c_str(),
                                    inplace?" (inplace)":"") ;
      out[0] = processPool.mexPull(tensorName, inplace) ;
      break ;

    case reset :
      (verbosity >= 2) && mexPrintf("vl_tflowmex: command 'reset'\n") ;
      processPool.shutdown() ; // gracefully (wait for others to finish)
      processPool.finalize() ; // no matter what
      break ;
  }
}

