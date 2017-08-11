// @file nnbnorm_gpu.cu
// @brief Batch normalization block GPU.
// @author Sebastien Ehrhardt
// @author Andrea Vedaldi

/*
Copyright (C) 2015-17 Sebastien Ehrhardt and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnbnorm.hpp"
#include "datacu.hpp"
#include "impl/blashelper.hpp"
#include "impl/sharedmem.cuh"
#include <cassert>
#include <cstdint>
#include <cfloat>

// -------------------------------------------------------------------
//                                                             Helpers
// -------------------------------------------------------------------
/*

# Overview

Batch normalization accumulates statistics for each feature channel
by summing across spatial locations and instances. Spatial locations
are contiguous in memory, but there is a gap when moving from an image
to the next.
 
The GPU runs in parallel blocks of threads (typically of 512 elements).
In an efficient implementation, the thread in a block operate in parallel
on blocks 512 consecutive elements, performing identical operations. For efficient
memory access, furthermore, the 512 memory blocks must be aligned.

Thus a thread block of sie bs should read consecutive blocks
of memory locations as follows:

0     bs    2bs   3bs
+block+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 
Howver, feature planes do not align to memory block boundaries and
there is a gaps when the next instance/image is visited.
The data that needs to be summed over looks like this:

0     bs    2bs   3bs
+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
    
    pstart   +psize               + 1pstride
    *--plane--*       -gap-       *---------*

We program the thread block to visit all the memory blocks necessary to
cover all the palens. As each block is applied to the data, some
threads may occasionally discard reads that are outside the required
range:
 
0     bs    2bs   3bs
+xxx--+-----+--xxx+     +     +xxx--+-----+--xxx+     +     +
    
    pstart   +psize               + 1pstride
    *---------*                   *---------*

We use a simple algorithm to peform this visit. Let p=0,1,... be the plane
index and b=0,1,... the blcok index.
 
1. Begin with p=0.
2. Find the first memory block b that overlaps with the plane p:
 
     b = pstart / bs.

3. Each thread in the thread block reads a certain element in memory block b.
   If this element is outside plane p it is ignored, otherwise it is accumulated.

4. If the block ends beyon the last element of the plane, i.e. if
 
     b * bs + bs >= p * pstride + pstart + psize,

   then the plane has been fully read: we increment p
   and continue form (2). Otherwise we increase b to read the next few
   elements of the current plane p.

This algorithm considers all planes in seuqence and visits each
one of their elements only once. Note that this works for any combination
of block size, plane offset, plane size, and plane stride. For example,
if planse are smaller than blocks, the same block will simply be
read multiple times, each time discarding different elements.

## Detailed block-plane association

In the scheme above, one thread block is responsible for accumulating
data for a single feature channel, across planes and instances/images.
Hence in these scheme numBlocks = numChannels. In pratice, it can
be preferable to increase the nubmer of blocks, particularly when
the batch is large, to allow more work to progress in parallel.
 
In order to do so, we use numBlocks = M numChannels, where M >= 1
is a multiplier. Thread block number tb operates on
 
   channel = tb % numChannels.

and on images
 
   (tb / numChannels) * M + [0 ... M-1],

In this manner, images are divided in numBlocksPerChannel = ceil(numImages / M),
and statistics are computed in parallel for each chunk.

# Reduction
 
Once all thread blocks are complete, partial sums for each feature
channel must be aggregated. This means summing at the level of:
 
1. Warps (up to 32 threads). A thread warp is a subst of highly coupled thread within
   a thread block. Threads are *coalesced* and
   run essentially in a single stream of vector instructions on the GPU,
   which also means that they stay syncrhonized implicitly. Threads
   in a warp write to the same shared memory area; this is reduced
   by performing a hierarchical summation.
 
2. Blocks (up to 512 threads). Thread blocks are assigned to a SM,
   and the SM breaks them down into warps for execution.
   Threads in the same block can be synchronised explicity using __syncthreads().
   They all run concurrently in the same SM and write to the same
   shared memory area like the warps. Reduction is also hierarchical,
   but must use __syncthreads().

3. Chunks: The numBlocksPerChannel partial results for each feature channel
   must be aggregated. This is done by storing partial results in a
   numChunk elements vector in global memory and running a GPU
   kernel to collapse it.

## Hierarchial reduction
 
This is used to accumualte a vector v[0], ..., v[blockSize-1] 
stored in the shared memory ara of a thread block.
 
This occurrs in stages, each time collapsing elements at a distance
blockSize/2^k. In particular, each thread t in the block does:
 
  t=0,...,blockSize/2^k:  v[t] = v[t] + v[t + blockSize/2^k].

Threads outside the active range do nothing. When k=log2(blockSize),
in particular, thread 0 does:
 
  t=0:  v[0] = v[0] + v[1]
 
which is the last summation in the reduction.
 
Every time the thread block performs a summation, the block must be
synchronized. There are two regimes:
 
1. When blockSize/2^k <=0 warpSize, snycrhonization is implicit as
   threads t only span a single warp.
 
2. When blockSize/2^k, we must ad __synchtreads() after
   the summation.

 
## Choosing the number of blocks

Each channel is processed by one or more blocks.
There are numBlocksPerChannel >= 1 blocks per channel, each working
on a subset of one or more images. There are

numBlocks = numBlocksPerChannel * numChannels

blocks in the grid.

We select numBlocksPerChannel to satisfy the following constraints:

1. There must be at least one block per channel:

      numBlocksPerChannel >= 1.

2. There must be at most one block per image:

      numBlocksPerChannel <= size.

3. The grid size must be less than 65536 (CUDA limit)

      numBlocksPerChannel <= 65536 / numChannels.

Note that constraints (1) and (3) can be satisfied only if
numChannels <= 65536. This is usually not a problem, but may fail
in general.

In general, (1--3) can be satisfied by setting numBlocksPerChannel=1.
However, this is suboptimal if there are too many operations
per block.

We would like to do at most

L = 10e3 * blockSize

operations per block and each block does

(planeArea * size)/numBlocksPerChannel

operation. Thus the target value for numBlocksPerChannel is

numBlocksPerChannel = ceil((planeArea * size) / L).
*/

// MSB_WARP = log2(WARP_SIZE)
#define WARP_SIZE 32
#define MSB_WARP 5

// macro function
#define min(a,b) (a > b ? b : a);

static inline int getBlockSize(int dataSize)
{
  int blockSize = VL_CUDA_NUM_THREADS / 2 ;
  if (dataSize < blockSize) {
    unsigned int numWarps = dataSize / WARP_SIZE ;
    if (numWarps < 4) {
      blockSize = 2 * WARP_SIZE ;
    }
    else if (numWarps < 8) {
      blockSize = 4 * WARP_SIZE ;
    }
    else {
      blockSize = 8 * WARP_SIZE ;
    }
  }
  return blockSize ;
}

// get the smallest x which is a multiple of factor
static inline int nextMultipleOf(int x, int factor)
{
  return factor * ((x + factor - 1)/factor) ;
}

template<typename T>
__forceinline__ __device__ void blockReduce(volatile T * mdata,
                                            unsigned int tid,
                                            unsigned int blockSize,
                                            unsigned int maxDataSize)
{
  // todo: get rid of maxDataSize?
  __syncthreads();
  if (blockSize >= 1024 && maxDataSize + WARP_SIZE >=512) { if (tid < 512) { mdata[tid] += mdata[tid + 512]; } __syncthreads(); } // mdata[0:511] = mdata[0:511] + mdata[512:1023]
  if (blockSize >= 512  && maxDataSize + WARP_SIZE >=256) { if (tid < 256) { mdata[tid] += mdata[tid + 256]; } __syncthreads(); } // mdata[0:255] = mdata[0:255] + mdata[256:511]
  if (blockSize >= 256  && maxDataSize + WARP_SIZE >=128) { if (tid < 128) { mdata[tid] += mdata[tid + 128]; } __syncthreads(); } // mdata[0:127] = mdata[0:127] + mdata[128:255]
  if (blockSize >= 128  && maxDataSize + WARP_SIZE >=64 ) { if (tid <  64) { mdata[tid] += mdata[tid + 64];  } __syncthreads(); } // mdata[0:63]  = mdata[0:63]  + mdata[64:127]
  if (tid < 32) {
    // now enter warp
    if (blockSize >=  64) { mdata[tid] += mdata[tid + 32]; } // mdata[0:31] = mdata[0:31] + mdata[32:63]
    if (blockSize >=  32) { mdata[tid] += mdata[tid + 16]; } // mdata[0:15] = mdata[0:15] + mdata[16:31]
    if (blockSize >=  16) { mdata[tid] += mdata[tid +  8]; } // mdata[0:7]  = mdata[0:7]  + mdata[7:15]
    if (blockSize >=   8) { mdata[tid] += mdata[tid +  4]; } // mdata[0:3]  = mdata[0:3]  + mdata[4:7]
    if (blockSize >=   4) { mdata[tid] += mdata[tid +  2]; } // mdata[0:1]  = mdata[0:1]  + mdata[2:3]
    if (blockSize >=   2) { mdata[tid] += mdata[tid +  1]; } // mdata[0]    = mdata[0]    + mdata[1]
  }
}

template<typename T>
__forceinline__ __device__ void blockReduce2(volatile T * mdata,
                                             volatile T * sdata,
                                             unsigned int tid,
                                             unsigned int blockSize,
                                             unsigned int maxDataSize)
{
  __syncthreads();
  if (blockSize >= 1024 && maxDataSize + WARP_SIZE >=512) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; mdata[tid] += mdata[tid + 512]; } __syncthreads(); }
  if (blockSize >= 512  && maxDataSize + WARP_SIZE >=256) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; mdata[tid] += mdata[tid + 256]; } __syncthreads(); }
  if (blockSize >= 256  && maxDataSize + WARP_SIZE >=128) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; mdata[tid] += mdata[tid + 128]; } __syncthreads(); }
  if (blockSize >= 128  && maxDataSize + WARP_SIZE >=64)  { if (tid <  64) { sdata[tid] += sdata[tid + 64];  mdata[tid] += mdata[tid + 64];  } __syncthreads(); }
  if (tid < 32) {
    if (blockSize >=  64) { sdata[tid] += sdata[tid + 32]; mdata[tid] += mdata[tid + 32]; }
    if (blockSize >=  32) { sdata[tid] += sdata[tid + 16]; mdata[tid] += mdata[tid + 16]; }
    if (blockSize >=  16) { sdata[tid] += sdata[tid +  8]; mdata[tid] += mdata[tid +  8]; }
    if (blockSize >=   8) { sdata[tid] += sdata[tid +  4]; mdata[tid] += mdata[tid +  4]; }
    if (blockSize >=   4) { sdata[tid] += sdata[tid +  2]; mdata[tid] += mdata[tid +  2]; }
    if (blockSize >=   2) { sdata[tid] += sdata[tid +  1]; mdata[tid] += mdata[tid +  1]; }
  }
}

template<typename T>
__forceinline__ __device__ void blockReduce4(volatile T * sdata,
                                             volatile T * mdata,
                                             volatile T * rdata,
                                             volatile T * tdata,
                                             unsigned int tid,
                                             unsigned int blockSize,
                                             unsigned int maxDataSize)
{
  __syncthreads();
  if (blockSize >= 1024 && maxDataSize + WARP_SIZE >= 512) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; mdata[tid] += mdata[tid + 512]; rdata[tid] += rdata[tid + 512]; tdata[tid] += tdata[tid + 512];} __syncthreads(); }
  if (blockSize >= 512 && maxDataSize + WARP_SIZE >= 256) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; mdata[tid] += mdata[tid + 256]; rdata[tid] += rdata[tid + 256]; tdata[tid] += tdata[tid + 256];} __syncthreads(); }
  if (blockSize >= 256 && maxDataSize + WARP_SIZE >= 128) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; mdata[tid] += mdata[tid + 128]; rdata[tid] += rdata[tid + 128]; tdata[tid] += tdata[tid + 128];} __syncthreads(); }
  if (blockSize >= 128 && maxDataSize + WARP_SIZE >= 64) { if (tid <  64) { sdata[tid] += sdata[tid + 64];  mdata[tid] += mdata[tid + 64];  rdata[tid] += rdata[tid + 64]; tdata[tid] += tdata[tid + 64];} __syncthreads(); }
  if (tid < 32) {
    if (blockSize >=  64) { sdata[tid] += sdata[tid + 32]; mdata[tid] += mdata[tid + 32]; rdata[tid] += rdata[tid + 32]; tdata[tid] += tdata[tid + 32];}
    if (blockSize >=  32) { sdata[tid] += sdata[tid + 16]; mdata[tid] += mdata[tid + 16]; rdata[tid] += rdata[tid + 16]; tdata[tid] += tdata[tid + 16];}
    if (blockSize >=  16) { sdata[tid] += sdata[tid +  8]; mdata[tid] += mdata[tid +  8]; rdata[tid] += rdata[tid +  8]; tdata[tid] += tdata[tid +  8];}
    if (blockSize >=   8) { sdata[tid] += sdata[tid +  4]; mdata[tid] += mdata[tid +  4]; rdata[tid] += rdata[tid +  4]; tdata[tid] += tdata[tid +  4];}
    if (blockSize >=   4) { sdata[tid] += sdata[tid +  2]; mdata[tid] += mdata[tid +  2]; rdata[tid] += rdata[tid +  2]; tdata[tid] += tdata[tid +  2];}
    if (blockSize >=   2) { sdata[tid] += sdata[tid +  1]; mdata[tid] += mdata[tid +  1]; rdata[tid] += rdata[tid +  1]; tdata[tid] += tdata[tid +  1];}
  }
}

// Get largest memory address that is aligned to a warp worth of T
// and smaller than x.

template<typename T>
__forceinline__ __device__ uintptr_t getBlockBeginning(void const * x)
{
  return (uintptr_t)(x) & (~((uintptr_t)(WARP_SIZE*sizeof(T)) - 1)) ;
}

// Use the current block of thread to sum over a given column of a matrix. The selected
// column is given by the thread block index in the block grid.
//
// This function uses an amoutn of scratch memory equal to blockSize*sizeof(T)
// where blockSize=blockDim.x.

template<typename T>
__forceinline__ __device__ T matrixSumHelper(T const * matrix, int numRows)
{
  // One thread block per column to sum
  // Shared memory is per-block, it holds blockSize intermediate reults
  //extern __shared__ T scratch [] ;
  SharedMemory<T> smem ;
  T * scratch = smem.getPointer() ;
  int tid = threadIdx.x ;
  int column = blockIdx.x ;
  int blockSize = blockDim.x ;

  // Note that scratch is different for different blocks, hence
  // matrix columns. Now fill scratch with partial sums, in a sliding-window
  // manner.
  scratch[tid] = 0 ;
  T const * columnBegin = matrix + column * numRows ;
  T const * columnEnd = columnBegin + numRows ;
  T const * block = (T const*) getBlockBeginning<T>(columnBegin) + tid ;
  while (block < columnEnd) {
    if (block >= columnBegin) {
      scratch[tid] += *block ;
    }
    block += blockSize ;
  }

  // Now scratch[] has blockSize partial sums for this column
  // Finish by reducing and saving
  blockReduce<T>(scratch, tid, blockSize, numRows) ;

  return scratch[0] ;
}

// This kernel accumulates means and variances for the data.
// Each block of thread sums over one or more data planes, resulting
// in an array accumulator[] of dimension numBlocksPerChannel x 2*numChannels.
//
// If each thread block scans all the images, then numBlocksPerChannel = 1.
// However, for efficiency different thread blocks do different
// subset of images, resulting in numBlocksPerChannel partial results to be summed
// later by a second kernel.
//
// The first part accumulator[:,0:numChannels-1] stores the data for the mean
// and the second part accumulator[:,numChannels,2*numChannels-1] the data
// for the sigmas.
//
// This function uses the sliding-window summing technique described
// above. It requires
//
//    2*sizeof(T)*blockSize
//
// bytes of shared scratch memory to hold to hold partial sums for
// means and sigmas.

template<typename T>
__global__ void accumulate_moments_partial(T * accumulator,
                                           T const * data,
                                           int planeArea,
                                           int numPlanes,
                                           int numChannels,
                                           int numBlocksPerChannel)
{
  int tid = threadIdx.x ;
  int plane = blockIdx.x ;
  int blockSize = blockDim.x ;
  int planeStride = gridDim.x ;
  int channel = blockIdx.x % numChannels ;

  //extern __shared__ T s [] ;
  SharedMemory<T> smem ;
  T * s = smem.getPointer() ;
  T * mdata = s ;
  T * sdata = mdata + blockSize ;

  mdata[tid] = 0 ;
  sdata[tid] = 0 ;

  while (plane < numPlanes) {
    T const * planeBegin = data + plane * planeArea ;
    T const * planeEnd = planeBegin + planeArea ;
    T const * block = (T const*) getBlockBeginning<T>(planeBegin) + tid ;
    while (block < planeEnd) {
      if (block >= planeBegin) {
        T x = *block ;
        mdata[tid] += x ;
        sdata[tid] += x * x ;
      }
      block += blockSize ;
    }
    plane += planeStride ;
  }

  blockReduce2<T>(sdata, mdata, tid, blockSize, planeArea) ;

  if (tid == 0) {
    int chunk = blockIdx.x / numChannels ;
    int i = chunk + channel * numBlocksPerChannel ;
    accumulator[i] = mdata[0];
    accumulator[i + gridDim.x] = sdata[0];
  }
}

// This kernel sums over the accumulator computed by the function
// above to obtain the moments.
//
// This kernel uses matrixSumHelper() defined above. Hence:
//
// 1. The block grid must be set to have a block
//    for each column of accumulator[]. There are here 2*numChannels columns.
//
// 2. There can be any (reasonable) blockSize. Blocks will iterate
//    over rows as needed to compte the operation.
//
// 3. It must be called with `blockSize*sizeof(T)` shared
//    scratch space.

template<typename T>
__global__ void accumulate_moments_finish(T * moments,
                                          T const * accumulator,
                                          int numRows)
{
  int tid = threadIdx.x ;
  int column = blockIdx.x ;
  T x = matrixSumHelper(accumulator, numRows) ;
  if (tid == 0) {
    moments[column] = x ;
  }
}

// After accumulation, we need to renormalize the moments.
//
// 1. It shoudl be called with enough threads to cover all
//    numChannels in the moments.
//
// 2. The actual number of blocks is determined based on the block
//    size to satisfy condition (2).

template<typename T>
__global__ void normalize_moments(T * moments,
                                  unsigned int numChannels,
                                  T mass,
                                  T epsilon)
{
  int unsigned i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i < numChannels){
    // max(0, __) is for numerical issues
    T mean = moments[i] / mass ;
    T sigma2 = max((T).0, moments[i + numChannels]/mass - mean*mean) ;
    moments[i] = mean ;
    moments[i + numChannels] = sqrt(sigma2 + epsilon);
  }
}

// Same as accumulate_moments above. Call with:
//
// 1. 2*sizeof(T)*blockSize scratch space
// 2.
//
// bytes of shared scratch memory to hold to hold partial sums for
// means and sigmas.
//
// Below, either accumulator is not NULL and derMultipliers, derBiases,
// and moments are, or the function is run in a `final' mode,
// with accumulator set to NULL, and the other points set to their
// `final' destination.

template<typename T>
__global__ void accumulate_ders_partial
(T * accumulator,
 T * derMultipliers,
 T * derBiases,
 T const * data,
 T const * derOutput,
 int planeArea,
 int numPlanes,
 int numChannels,
 int numBlocksPerChannel)
{
  int tid = threadIdx.x ;
  int plane = blockIdx.x ;
  int blockSize = blockDim.x ;
  int planeStride = gridDim.x ;
  int channel = blockIdx.x % numChannels ;
  //extern __shared__ T s[] ;
  SharedMemory<T> smem ;
  T * s = smem.getPointer() ;

  T * mdata = s ;
  T * sdata = mdata + blockSize ;

  mdata[tid] = 0 ;
  sdata[tid] = 0 ;

  while (plane < numPlanes) {
    T const * planeBegin = data + plane * planeArea ;
    T const * planeEnd = planeBegin + planeArea ;
    T const * block = (T const*) getBlockBeginning<T>(planeBegin) + tid ;
    T const * dblock = derOutput + (block - data) ;
    while (block < planeEnd) {
      if (block >= planeBegin) {
        T x = *block ;
        T dy = *dblock ;
        mdata[tid] += x * dy ;
        sdata[tid] += dy ;
      }
      block += blockSize ;
      dblock += blockSize ;
    }
    plane += planeStride ;
  }

  blockReduce2<T>(sdata, mdata, tid, blockSize, planeArea);

  if (tid == 0) {
    if (numChannels == gridDim.x) {
      // Final output ready
      derMultipliers[blockIdx.x] = mdata[0];
      derBiases[blockIdx.x] = sdata[0];
    } else {
      // Partially accumulated outut
      int chunk = blockIdx.x / numChannels ;
      int i = chunk + channel * numBlocksPerChannel ;
      accumulator[i] = mdata[0]; // derMultipliers
      accumulator[i + gridDim.x] = sdata[0]; // derBiases
    }
  }
}

template<typename T>
__global__ void accumulate_ders_finish(T * derMultipliers,
                                       T * derBiases,
                                       T const * accumulator,
                                       int numBlocksPerChannel,
                                       int numChannels)
{
  int tid = threadIdx.x ;
  int column = blockIdx.x ;
  T x = matrixSumHelper(accumulator, numBlocksPerChannel) ;
  if (tid == 0) {
    // Recall that the matrix stores in order [derMultipliers derBiases means sigmas]
    // containing four types of data
    int type = column / numChannels ;
    int channel = column % numChannels ;

    if (type == 0) {
      derMultipliers[channel] = x ;
    }
    else {
      derBiases[channel] = x ;
    }
  }
}

template<typename T>
__global__ void normalize_ders(T * derMultipliers,
                               T const * derBiases,
                               T const * moments,
                               unsigned int numChannels,
                               T mass,
                               T epsilon)
{
  unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx < numChannels){
    T mean = moments[idx] ;
    T sigma = moments[idx + numChannels] ;
    derMultipliers[idx] = (derMultipliers[idx] - mean*derBiases[idx]) / sigma ;
  }
}

// Same as accumulate_moments above. Call with:
//
// 1. 4*sizeof(T)*blockSize scratch space
// 2.
//
// bytes of shared scratch memory to hold to hold partial sums for
// means and sigmas.
//
// Below, either accumulator is not NULL and derMultipliers, derBiases,
// and moments are, or the function is run in a `final' mode,
// with accumulator set to NULL, and the other points set to their
// `final' destination.

template<typename T>
__global__ void accumulate_ders_and_moments_partial
(T * accumulator,
 T * derMultipliers,
 T * derBiases,
 T * moments,
 T const * data,
 T const * derOutput,
 int planeArea,
 int numPlanes,
 int numChannels,
 int numBlocksPerChannel)
{
  int tid = threadIdx.x ;
  int plane = blockIdx.x ;
  int blockSize = blockDim.x ;
  int planeStride = gridDim.x ;
  int channel = blockIdx.x % numChannels ;
  //extern __shared__ T s[] ;
  SharedMemory<T> smem ;
  T * s = smem.getPointer() ;

  T * mdata = s ;
  T * sdata = mdata + blockSize ;
  T * rdata = sdata + blockSize ;
  T * tdata = rdata + blockSize ;

  mdata[tid] = 0 ;
  sdata[tid] = 0 ;
  rdata[tid] = 0 ;
  tdata[tid] = 0 ;

  while (plane < numPlanes) {
    T const * planeBegin = data + plane * planeArea ;
    T const * planeEnd = planeBegin + planeArea ;
    T const * block = (T const*) getBlockBeginning<T>(planeBegin) + tid ;
    T const * dblock = derOutput + (block - data) ;
    while (block < planeEnd) {
      if (block >= planeBegin) {
        T x = *block ;
        T dy = *dblock ;
        mdata[tid] += x * dy ;
        sdata[tid] += dy ;
        rdata[tid] += x * x ;
        tdata[tid] += x ;
      }
      block += blockSize ;
      dblock += blockSize ;
    }
    plane += planeStride ;
  }

  blockReduce4<T>(sdata, mdata, rdata, tdata, tid, blockSize, planeArea);

  if (tid == 0) {
    if (numChannels == gridDim.x) {
      // Final output ready
      derMultipliers[blockIdx.x] = mdata[0];
      derBiases[blockIdx.x] = sdata[0];
      moments[blockIdx.x] = tdata[0];
      moments[blockIdx.x+numChannels] = rdata[0];
    } else {
      // Partially accumulated outut
      int chunk = blockIdx.x / numChannels ;
      int i = chunk + channel * numBlocksPerChannel ;
      accumulator[i] = mdata[0]; // derMultipliers
      accumulator[i + gridDim.x] = sdata[0]; // derBiases
      accumulator[i + 2*gridDim.x] = tdata[0]; // means
      accumulator[i + 3*gridDim.x] = rdata[0]; // sigmas
    }
  }
}

template<typename T>
__global__ void accumulate_ders_and_moments_finish(T * derMultipliers,
                                                   T * derBiases,
                                                   T * moments,
                                                   T const * accumulator,
                                                   int numBlocksPerChannel,
                                                   int numChannels)
{
  int tid = threadIdx.x ;
  int column = blockIdx.x ;
  T x = matrixSumHelper(accumulator, numBlocksPerChannel) ;
  if (tid == 0) {
    // Recall that the matrix stores in order [derMultipliers derBiases means sigmas]
    // containing four types of data
    int type = column / numChannels ;
    int channel = column % numChannels ;

    if (type == 0) {
      derMultipliers[channel] = x ;
    }
    else if (type == 1) {
      derBiases[channel] = x ;
    }
    else if (type == 2) {
      moments[channel] = x ;
    }
    else {
      moments[channel + numChannels] = x ;
    }
  }
}

template<typename T>
__global__ void normalize_ders_and_moments(T * derMultipliers,
                                           T * derBiases,
                                           T * moments,
                                           unsigned int numChannels,
                                           T mass,
                                           T epsilon)
{
  unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx < numChannels){
    T mean = moments[idx] / mass;
    T sigma2 = max((T).0, moments[idx + numChannels]/mass - mean*mean) ;
    T sigma = sqrt(sigma2 + epsilon);
    moments[idx] = mean ;
    moments[idx + numChannels] = sigma ;
    derMultipliers[idx] = (derMultipliers[idx]-mean*derBiases[idx]) / sigma ;
  }
}

// Call this kernel like compute_moments, but it does not need a scratch sapce

template<typename T>
__global__ void batch_normalize_forward(T * outputData,
                                        T const * moments,
                                        T const * data,
                                        T const * multipliers,
                                        T const * biases,
                                        int planeArea,
                                        int numPlanes,
                                        int numChannels)
{
  int tid = threadIdx.x ;
  int plane = blockIdx.x ;
  int blockSize = blockDim.x ;
  int planeStride = gridDim.x ;
  int channel = blockIdx.x % numChannels ;

  T mean = moments[channel];
  T sigma = moments[channel+numChannels];
  T multiplier = multipliers[channel];
  T bias = biases[channel];
  T coefficient = multiplier / sigma ;

  while (plane < numPlanes) {
    T const * planeBegin = data + plane * planeArea ;
    T const * planeEnd = planeBegin + planeArea ;
    T const * block = (T const*) getBlockBeginning<T>(planeBegin) + tid ;
    T * oblock = outputData + (block - data) ;
    while (block < planeEnd) {
      if (block >= planeBegin) {
        *oblock = coefficient * (*block - mean) + bias ;
      }
      block += blockSize ;
      oblock += blockSize ;
    }
    plane += planeStride ;
  }
}

template<typename T>
__global__ void batch_normalize_backward(T * derData,
                                         T const * moments,
                                         T const * data,
                                         T const * multipliers,
                                         T const * derMultipliers,
                                         T const * derBiases,
                                         T const * derOutput,
                                         int planeArea,
                                         int numPlanes,
                                         int numChannels,
                                         T mass)
{
  int tid = threadIdx.x ;
  int plane = blockIdx.x ;
  int blockSize = blockDim.x ;
  int planeStride = gridDim.x ;
  int channel = blockIdx.x % numChannels ;

  T mu = moments[channel];
  T sigma = moments[channel + numChannels] ;
  T multiplier = multipliers[channel] ;
  T derMultiplier = derMultipliers[channel] ;

  T muz = derBiases[channel] / mass;
  T G1 = multiplier / sigma ;
  T G2 = G1 * derMultiplier / (mass*sigma);

  while (plane < numPlanes) {
    T const * planeBegin = data + plane * planeArea ;
    T const * planeEnd = planeBegin + planeArea ;
    T const * block = (T const*) getBlockBeginning<T> (planeBegin) + tid ;
    T const * dblock = derOutput + (block - data) ;
    T * oblock = derData + (block - data) ;
    while (block < planeEnd) {
      if (block >= planeBegin) {
        *oblock = G1 * (*dblock - muz) - G2 * (*block - mu);
      }
      block += blockSize ;
      dblock += blockSize ;
      oblock += blockSize ;
    }
    plane += planeStride ;
  }
}

// -------------------------------------------------------------------
//                                                             Forward
// -------------------------------------------------------------------

template<DataType dataType>
struct BatchNormForwardWithMoment<VLDT_GPU, dataType>
{
  vl::ErrorCode operator()(BatchNorm &op,
                           Tensor &output,
                           Tensor const &moment,
                           Tensor const &input,
                           Tensor const &multiplier,
                           Tensor const &bias)
  {
    cudaError_t status ;
    typedef typename vl::DataTypeTraits<dataType>::type type ;
    auto height = input.getHeight() ;
    auto width = input.getWidth() ;
    auto numChannels = input.getDepth() ;
    auto size = input.getSize() ;
    auto outputData = (type*)output.getMemory() ;
    auto momentData = (type const*)moment.getMemory() ;
    auto inputData = (type const*)input.getMemory() ;
    auto multiplierData = (type const*)multiplier.getMemory() ;
    auto biasData = (type const*)bias.getMemory() ;

    size_t planeArea = height * width ;
    size_t numPlanes = numChannels * size ;

    // Compute number compute chunks.
    size_t blockSize = getBlockSize(planeArea) ;
    //size_t L = 10000 * blockSize ;
    //size_t numBlocksPerChannel = (planeArea * size + L - 1) / L ;
    //numBlocksPerChannel = std::min(numBlocksPerChannel, size) ;
    //numBlocksPerChannel = std::min(numBlocksPerChannel, 65536 / numChannels) ;
    //numBlocksPerChannel = std::max(numBlocksPerChannel, 1) ;
    size_t numBlocksPerChannel = 1  ;
    size_t numBlocks = numChannels * numBlocksPerChannel ;
    assert(numBlocksPerChannel >= 1) ;
    assert(numBlocksPerChannel <= size) ;
    assert(numBlocks <= 65536) ;

    batch_normalize_forward <<<numBlocks, blockSize>>>
    (outputData, momentData, inputData, multiplierData, biasData,
     planeArea, numPlanes, numChannels) ;

    status = cudaPeekAtLastError() ;
    return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
  }
} ;

template<DataType dataType>
struct BatchNormForward<VLDT_GPU, dataType>
{
  vl::ErrorCode operator()(BatchNorm &op,
                           Tensor &output,
                           Tensor &moment,
                           Tensor const &input,
                           Tensor const &multiplier,
                           Tensor const &bias)
  {
    cudaError_t status ;
    typedef typename vl::DataTypeTraits<dataType>::type type ;
    auto height = input.getHeight() ;
    auto width = input.getWidth() ;
    auto numChannels = input.getDepth() ;
    auto size = input.getSize() ;
    auto outputData = (type*)output.getMemory() ;
    auto inputData = (type const*)input.getMemory() ;
    auto multiplierData = (type const*)multiplier.getMemory() ;
    auto biasData = (type const*)bias.getMemory() ;

    size_t planeArea = height * width ;
    size_t numPlanes = numChannels * size ;

    // Compute number compute chunks.
    size_t blockSize = getBlockSize(planeArea) ;
    //size_t L = 10000 * blockSize ;
    //size_t numBlocksPerChannel = (planeArea * size + L - 1) / L ;
    //numBlocksPerChannel = min(numBlocksPerChannel, size) ;
    //numBlocksPerChannel = min(numBlocksPerChannel, 65536 / numChannels) ;
    //numBlocksPerChannel = max(numBlocksPerChannel, 1) ;
    size_t numBlocksPerChannel = 1  ;
    size_t numBlocks = numChannels * numBlocksPerChannel ;

    // Get scratch space.
    size_t accumulatorSize = (numBlocksPerChannel == 1) ? 0 : 2 * nextMultipleOf(numBlocks, WARP_SIZE) ;
    size_t workspaceSize = accumulatorSize + (moment.getMemory() ? 0 : 2 * numChannels) ;
    type * workspace = (type*)op.context.getWorkspace(vl::VLDT_GPU, workspaceSize * sizeof(type)) ;
    if (workspace == NULL && workspaceSize > 0) {
      return VLE_OutOfMemory ;
    }
    type * accumulatorData = workspace ;


    Tensor ownMoment(moment) ;
    if (ownMoment.getMemory() == NULL) {
      ownMoment.setMemory(workspace + accumulatorSize) ;
    }
    auto momentData = (type*)ownMoment.getMemory() ;

    // Accumulate moments.
    if (numBlocksPerChannel > 1) {
      // Partial.
      accumulate_moments_partial <<<numBlocks, blockSize, 2*blockSize*sizeof(type)>>>
      (accumulatorData,
       inputData,
       planeArea,
       numPlanes,
       numChannels,
       numBlocksPerChannel) ;

      status = cudaPeekAtLastError() ;
      if (status != cudaSuccess) return vl::VLE_Cuda ;

      // Total.
      int blockSizeForSum = getBlockSize(numBlocksPerChannel) ;
      accumulate_moments_finish <<<2*numChannels, blockSizeForSum, blockSizeForSum*sizeof(type)>>>
      (momentData, accumulatorData, numBlocksPerChannel) ;
      status = cudaPeekAtLastError() ;
      if (status != cudaSuccess) return vl::VLE_Cuda ;
    } else {
      // Total directly.
      accumulate_moments_partial <<<numBlocks, blockSize, 2*blockSize*sizeof(type)>>>
      (momentData,
       inputData,
       planeArea,
       numPlanes,
       numChannels,
       1) ;
    }

    // Normalize moments.
    type mass = planeArea*size;
    normalize_moments <<<divideAndRoundUp(numChannels,blockSize),blockSize>>>
    (momentData, numChannels, mass, (type)op.epsilon) ;

    // Normalize the data and apply multipliers and bias.
    batch_normalize_forward <<<numBlocks, blockSize>>>
    (outputData,
     momentData, inputData, multiplierData, biasData,
     planeArea,
     numPlanes,
     numChannels) ;

    status = cudaPeekAtLastError() ;
    return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
  }
} ;

// -------------------------------------------------------------------
//                                                            Backward
// -------------------------------------------------------------------

template<DataType dataType>
struct BatchNormBackwardWithMoment<VLDT_GPU, dataType>
{
  vl::ErrorCode operator()(BatchNorm &op,
                           Tensor &derInput,
                           Tensor &derMultiplier,
                           Tensor &derBias,
                           Tensor const &moment,
                           Tensor const &input,
                           Tensor const &multiplier,
                           Tensor const &bias,
                           Tensor const &derOutput)
  {
    cudaError_t status ;
    typedef typename vl::DataTypeTraits<dataType>::type type ;
    auto height = input.getHeight() ;
    auto width = input.getWidth() ;
    auto numChannels = input.getDepth() ;
    auto size = input.getSize() ;
    auto derInputData = (type*)derInput.getMemory() ;
    auto derBiasData = (type*)derBias.getMemory() ;
    auto derMultiplierData = (type*)derMultiplier.getMemory() ;
    auto inputData = (type const*)input.getMemory() ;
    auto momentData = (type const*)moment.getMemory() ;
    auto multiplierData = (type const*)multiplier.getMemory() ;
    auto biasData = (type const*)bias.getMemory() ;
    auto derOutputData = (type const*)derOutput.getMemory() ;

    size_t planeArea = height * width ;
    size_t numPlanes = numChannels * size ;

    // Compute number compute chunks.
    size_t blockSize = getBlockSize(planeArea) ;
    //size_t L = 10000 * blockSize ;
    //size_t numBlocksPerChannel = (planeArea * size + L - 1) / L ;
    //numBlocksPerChannel = std::min(numBlocksPerChannel, size) ;
    //numBlocksPerChannel = std::min(numBlocksPerChannel, 65536 / numChannels) ;
    //numBlocksPerChannel = std::max(numBlocksPerChannel, 1) ;
    size_t numBlocksPerChannel = 1  ;
    size_t numBlocks = numChannels * numBlocksPerChannel ;

    // Mean, variance, derMultiplier and derBias computation.
    if (numBlocksPerChannel > 1) {

      // Get scratch space.
      size_t workspaceSize = 2 * nextMultipleOf(numBlocks, WARP_SIZE) ;
      type * accumulatorData = (type*)op.context.getWorkspace
      (vl::VLDT_GPU, workspaceSize * sizeof(type)) ;
      if (accumulatorData == 0) {
        return VLE_OutOfMemory ;
      }

      // Partial.
      accumulate_ders_partial<type> <<<numBlocks, blockSize, 2*blockSize*sizeof(type)>>>
      (accumulatorData,
       NULL, NULL,
       inputData,
       derOutputData,
       planeArea,
       numPlanes,
       numChannels,
       numBlocksPerChannel) ;

      status = cudaPeekAtLastError() ;
      if (status != cudaSuccess) return vl::VLE_Cuda ;

      // Total.
      int blockSizeSum = getBlockSize(numBlocksPerChannel) ;
      accumulate_ders_finish<type> <<<2*numChannels, blockSizeSum, blockSizeSum*sizeof(type)>>>
      (derMultiplierData, derBiasData, accumulatorData, numBlocksPerChannel, numChannels) ;

      status = cudaPeekAtLastError() ;
      if (status != cudaSuccess) return vl::VLE_Cuda ;
    }
    else {
      // Total.
      accumulate_ders_partial<type> <<<numBlocks, blockSize, 2*blockSize*sizeof(type)>>>
      (NULL,
       derMultiplierData, derBiasData, inputData, derOutputData,
       planeArea,
       numPlanes,
       numChannels,
       1) ;

      status = cudaPeekAtLastError() ;
      if (status != cudaSuccess) return vl::VLE_Cuda ;
    }

    // Normalize derMultiplier and derBias.
    type mass = planeArea*size;
    normalize_ders<type> <<<divideAndRoundUp(numChannels,blockSize),blockSize>>>
    (derMultiplierData, derBiasData, momentData, numChannels, mass, op.epsilon) ;

    // Compute input derivative.
    batch_normalize_backward<type> <<<numBlocks, blockSize>>>
    (derInputData, momentData, inputData, multiplierData,
     derMultiplierData, derBiasData, derOutputData,
     planeArea, numPlanes, numChannels,
     mass) ;

    status = cudaPeekAtLastError() ;
    if (status != cudaSuccess) return vl::VLE_Cuda ;

    return VLE_Success ;
  }
} ;


template<DataType dataType>
struct BatchNormBackward<VLDT_GPU, dataType>
{
  vl::ErrorCode operator()(BatchNorm &op,
                           Tensor &derInput,
                           Tensor &derMultiplier,
                           Tensor &derBias,
                           Tensor &moment,
                           Tensor const &input,
                           Tensor const &multiplier,
                           Tensor const &bias,
                           Tensor const &derOutput)
  {
    cudaError_t status ;
    typedef typename vl::DataTypeTraits<dataType>::type type ;
    auto height = input.getHeight() ;
    auto width = input.getWidth() ;
    auto numChannels = input.getDepth() ;
    auto size = input.getSize() ;
    auto derInputData = (type*)derInput.getMemory() ;
    auto derBiasData = (type*)derBias.getMemory() ;
    auto derMultiplierData = (type*)derMultiplier.getMemory() ;
    auto inputData = (type const*)input.getMemory() ;
    auto multiplierData = (type const*)multiplier.getMemory() ;
    auto biasData = (type const*)bias.getMemory() ;
    auto derOutputData = (type const*)derOutput.getMemory() ;

    size_t planeArea = height * width ;
    size_t numPlanes = numChannels * size ;

    // Compute number compute chunks.
    size_t blockSize = getBlockSize(planeArea) ;
    //size_t L = 10000 * blockSize ;
    //size_t numBlocksPerChannel = (planeArea * size + L - 1) / L ;
    //numBlocksPerChannel = min(numBlocksPerChannel, size) ;
    //numBlocksPerChannel = min(numBlocksPerChannel, 65536 / numChannels) ;
    //numBlocksPerChannel = max(numBlocksPerChannel, 1) ;
    size_t numBlocksPerChannel = 1  ;
    size_t numBlocks = numChannels * numBlocksPerChannel ;

    // Get scratch space.
    size_t accumulatorSize = (numBlocksPerChannel == 1) ? 0 : 4 * nextMultipleOf(numBlocks, WARP_SIZE) ;
    size_t workspaceSize = accumulatorSize + (moment.getMemory() ? 0 : 2 * numChannels) ;
    type * workspace = (type*)op.context.getWorkspace(vl::VLDT_GPU, workspaceSize * sizeof(type)) ;
    type * accumulatorData = workspace ;
    if (workspace == NULL) {
      return VLE_OutOfMemory ;
    }

    Tensor ownMoment(moment) ;
    if (ownMoment.getMemory() == NULL) {
      ownMoment.setMemory(workspace + accumulatorSize) ;
    }
    auto momentData = (type*)ownMoment.getMemory() ;

    // Mean, variance, derMultiplier and derBias computation.
    if (numBlocksPerChannel > 1) {
      // Partial.
      accumulate_ders_and_moments_partial<type> <<<numBlocks, blockSize, 4*blockSize*sizeof(type)>>>
      (accumulatorData,
       NULL, NULL, NULL,
       inputData,
       derOutputData,
       planeArea,
       numPlanes,
       numChannels,
       numBlocksPerChannel) ;

      status = cudaPeekAtLastError() ;
      if (status != cudaSuccess) return vl::VLE_Cuda ;

      // Total.
      int blockSizeSum = getBlockSize(numBlocksPerChannel) ;
      accumulate_ders_and_moments_finish<type> <<<4*numChannels, blockSizeSum, blockSizeSum*sizeof(type)>>>
      (derMultiplierData, derBiasData, momentData, accumulatorData, numBlocksPerChannel, numChannels) ;

      status = cudaPeekAtLastError() ;
      if (status != cudaSuccess) return vl::VLE_Cuda ;
    }
    else {
      // Total.
      accumulate_ders_and_moments_partial<type> <<<numBlocks, blockSize, 4*blockSize*sizeof(type)>>>
      (NULL,
       derMultiplierData, derBiasData, momentData,
       inputData, derOutputData,
       planeArea,
       numPlanes,
       numChannels,
       1) ;

      status = cudaPeekAtLastError() ;
      if (status != cudaSuccess) return vl::VLE_Cuda ;
    }

    // Normalize derMultiplier and derBias.
    type mass = planeArea*size;
    normalize_ders_and_moments<type> <<<divideAndRoundUp(numChannels,blockSize),blockSize>>>
    (derMultiplierData, derBiasData, momentData, numChannels, mass, op.epsilon) ;

    // Compute derInput.
    batch_normalize_backward<type> <<<numBlocks, blockSize>>>
    (derInputData,
     momentData, inputData,
     multiplierData, derMultiplierData, derBiasData, derOutputData,
     planeArea, numPlanes, numChannels,
     mass) ;

    status = cudaPeekAtLastError() ;
    if (status != cudaSuccess) return vl::VLE_Cuda ;

    return VLE_Success ;
  }
} ;
