#ifndef ENABLE_GPU
#error "nnconv_subsample_gpu.cu cannot be compiled without GPU support"
#endif

// -------------------------------------------------------------------
//                                                             Helpers
// -------------------------------------------------------------------

template<typename T> __global__ void
subsample_forward_kernel
(T* output,
 const T* data,
 const int outputHeight,
 const int outputWidth,
 const int outputVolume,
 const int height,
 const int width,
 const int strideY,
 const int strideX,
 const int padTop,
 const int padLeft)
{
  int outputIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (outputIndex < outputVolume) {
    /* outputIndex = x
     + y * outputWidth
     + z * (outputWidth * outputHeight) ;
     */
    int py = outputIndex ;
    int px = py / outputHeight ;
    int channel = px / outputWidth ;
    px %= outputWidth ;
    py %= outputHeight ;
    int x1 = px * strideX - padLeft ;
    int y1 = py * strideY - padTop ;
    data += channel * (width*height) ;
    T value = 0 ;
    if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
      value = data[x1 * height + y1] ;
    }
    output[outputIndex] =  value ;
  }
}

template<typename T>
__global__ void subsample_backward_kernel
(T* derData,
 const T* derOutput,
 const int outputHeight,
 const int outputWidth,
 const int dataVolume,
 const int height,
 const int width,
 const int strideY,
 const int strideX,
 const int padTop,
 const int padLeft)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < dataVolume) {
    int y = index ;
    int x = y / height ;
    int channel = x / width ;
    x %= width ;
    y %= height ;
    derOutput += channel * outputHeight * outputWidth ;
    int px = (x + padLeft) / strideX ;
    int py = (y + padTop) / strideY ;
    if (x == strideX * px - padLeft &&
        y == strideY * py - padTop) {
      derData[index] = derOutput[px * outputHeight + py] ;
    } else {
      derData[index] = 0 ;
    }
  }
}
// -------------------------------------------------------------------
//                                                              Driver
// -------------------------------------------------------------------

template<vl::DataType dataType>
struct SubsampleForward<vl::VLDT_GPU, dataType>
{
  vl::ErrorCode operator()(Convolution const &op,
                           Tensor &output,
                           Tensor const &input)
  {
    static const std::string signature = std::string("SubsampleForward[MCN,")
    + DeviceTypeTraits<VLDT_GPU>::name + "," + DataTypeTraits<dataType>::name + "]" ;
    VLLOG(op,1) << signature ;

    typedef typename vl::DataTypeTraits<dataType>::type type ;
    Int outputVolume = output.getNumElements() ;

    subsample_forward_kernel<type>
    <<< divideAndRoundUp((unsigned)outputVolume,VL_CUDA_NUM_THREADS),VL_CUDA_NUM_THREADS >>>
    ((type*)output.getMemory(),
     (type const*)input.getMemory(),
     (int)output.getHeight(), (int)output.getWidth(), (int)output.getNumElements(),
     (int)input.getHeight(), (int)input.getWidth(),
     (int)op.getStride(0), (int)op.getStride(1),
     (int)op.getPadding(0), (int)op.getPadding(2));

    return op.getContext().setError(op.getContext().getCudaHelper().catchCudaError(signature.c_str())) ;
  }
} ;


template<vl::DataType dataType>
struct SubsampleBackward<vl::VLDT_GPU, dataType>
{
  vl::ErrorCode operator()(Convolution const &op,
                           Tensor &derInput,
                           Tensor const &derOutput)
  {
    static const std::string signature = std::string("SubsampleBackward[MCN,")
    + DeviceTypeTraits<VLDT_GPU>::name + "," + DataTypeTraits<dataType>::name + "]" ;

    VLLOG(op,1) << signature ;

    typedef typename vl::DataTypeTraits<dataType>::type type ;
    Int volume = derInput.getNumElements() ;

    subsample_backward_kernel<type>
    <<< divideAndRoundUp((unsigned)volume,VL_CUDA_NUM_THREADS),VL_CUDA_NUM_THREADS >>>
    ((type*)derInput.getMemory(),
     (type const*)derOutput.getMemory(),
     (int)derOutput.getHeight(), (int)derOutput.getWidth(), (int)volume,
     (int)derInput.getHeight() , (int)derInput.getWidth() ,
     (int)op.getStride(0), (int)op.getStride(1),
     (int)op.getPadding(0), (int)op.getPadding(2)) ;

    return op.getContext().setError(op.getContext().getCudaHelper().catchCudaError(signature.c_str())) ;
  }
} ;

