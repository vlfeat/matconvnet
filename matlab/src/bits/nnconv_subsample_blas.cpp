// -------------------------------------------------------------------
//                                                             Forward
// -------------------------------------------------------------------

template<vl::DataType dataType>
struct SubsampleForward<vl::VLDT_CPU, dataType>
{
  vl::ErrorCode operator()(Convolution const &op,
                           Tensor &output,
                           Tensor const &input)
  {
    static const std::string signature = std::string("SubsampleForward[MCN,")
    + DeviceTypeTraits<VLDT_CPU>::name + "," + DataTypeTraits<dataType>::name + "]" ;

    VLLOG(op,1) << signature ;

    typedef typename vl::DataTypeTraits<dataType>::type type ;
    Int width = input.getWidth() ;
    Int height = input.getHeight() ;
    Int depth = input.getNumChannels() ;
    Int size = input.getCardinality() ;
    auto inputData = (type*)input.getMemory() ;
    auto outputData = (type*)output.getMemory() ;

    Int outputHeight = output.getHeight() ;
    Int outputWidth = output.getWidth() ;
    Int strideY = op.getStride(0) ;
    Int strideX = op.getStride(1) ;
    Int padTop = op.getPadding(0) ;
    Int padLeft = op.getPadding(2) ;

    for (Int z = 0; z < depth * size ; ++z) {
      for (Int x = 0; x < outputWidth ; ++x) {
        for (Int y = 0; y < outputHeight ; ++y) {
          auto x1 = x * strideX - padLeft ;
          auto y1 = y * strideY - padTop ;
          type value = 0 ;
          if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
            value = inputData[x1 * height + y1] ;
          }
          outputData[x * outputHeight + y] = value ;
        }
      }
      inputData += width*height ;
      outputData += outputWidth*outputHeight ;
    }
    return VLE_Success ;
  }
} ;

template<vl::DeviceType deviceType, vl::DataType dataType>
struct SubsampleAndBiasForward
{
  vl::ErrorCode operator()(Convolution const &op,
                           Tensor &output,
                           Tensor const &input,
                           Tensor const &biases)
  {

    static const std::string signature = std::string("SubsampleForward[BLAS,")
    + DeviceTypeTraits<deviceType>::name + "," + DataTypeTraits<dataType>::name + "]" ;

    VLLOG(op,1) << signature ;

    vl::ErrorCode error ;
    typedef typename vl::DataTypeTraits<dataType>::type type ;

    error = SubsampleForward<deviceType,dataType>()(op,output,input) ;
    if (error != VLE_Success) { return error ; }

    auto numOutputPixels = output.getHeight() * output.getWidth() ;
    type const* allOnesMemory = (type*) op.getContext().getAllOnes(deviceType, dataType, (size_t)numOutputPixels) ;

    if (allOnesMemory == NULL) {
      error = op.getContext().getLastError() ;
      goto done ;
    }

    for (Int image = 0 ; image < input.getCardinality() ; ++image) {
      auto outputOffset = (output.getHeight()*output.getWidth()*output.getNumChannels()) * image ;
      if (biases) {
        type alpha = 1 ;
        type beta = 1 ;
        error = vl::impl::blas<deviceType, dataType>::gemm
        (op.getContext(),
         'n', 'n',
         numOutputPixels,
         biases.getNumElements(), 1,
         alpha,
         allOnesMemory, numOutputPixels,
         (type*)biases.getMemory(), 1,
         beta,
         (type*)output.getMemory() + outputOffset, numOutputPixels) ;
        if (error != vl::VLE_Success) { goto done ; }
      }
    }
  done:
    return op.getContext().passError(error,signature.c_str()) ;
  }
} ;

// -------------------------------------------------------------------
//                                                            Backward
// -------------------------------------------------------------------

template<vl::DataType dataType>
struct SubsampleBackward<vl::VLDT_CPU, dataType>
{
  vl::ErrorCode operator()(Convolution const &op,
                           Tensor &derInput,
                           Tensor const &derOutput)
  {
    static const std::string signature = std::string("SubsampleBackward[MCN,")
    + DeviceTypeTraits<VLDT_CPU>::name + "," + DataTypeTraits<dataType>::name + "]" ;

    VLLOG(op,1) << signature ;

    assert(derInput) ;
    assert(derOutput) ;

    typedef typename vl::DataTypeTraits<dataType>::type type ;
    auto width = derInput.getWidth() ;
    auto height = derInput.getHeight() ;
    auto depth = derInput.getNumChannels() ;
    auto size = derInput.getCardinality() ;
    auto derInputData = (type*)derInput.getMemory() ;
    auto derOutputData = (type*)derOutput.getMemory() ;

    Int outputHeight = derOutput.getHeight() ;
    Int outputWidth = derOutput.getWidth() ;
    Int strideY = op.getStride(0) ;
    Int strideX = op.getStride(1) ;
    Int padTop = op.getPadding(0) ;
    Int padLeft = op.getPadding(2) ;

    memset(derInputData, 0, sizeof(type) * size_t(width * height * depth * size)) ;

    for (Int z = 0; z < depth * size; ++z) {
      for (Int px = 0; px < outputWidth; ++px) {
        for (Int py  = 0; py < outputHeight; ++py) {
          auto x1 = px * strideX - padLeft ;
          auto y1 = py * strideY - padTop ;
          if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
            derInputData[x1 * height + y1]
            = derOutputData[px * outputHeight + py] ;
          }
        }
      }
      derInputData += width*height ;
      derOutputData += outputWidth*outputHeight ;
    }
    return VLE_Success ;
  }
} ;

template<vl::DeviceType deviceType, vl::DataType dataType>
struct SubsampleAndBiasBackward
{
  vl::ErrorCode operator()(Convolution const &op,
                           Tensor derInput,
                           Tensor derBiases,
                           Tensor derOutput)
  {
    static const std::string signature = std::string("SubsampleAndBiasBackward[BLAS,")
    + DeviceTypeTraits<deviceType>::name + "," + DataTypeTraits<dataType>::name + "]" ;

    VLLOG(op,1) << signature ;

    assert(derOutput) ;

    vl::ErrorCode error = VLE_Success ;
    typedef typename vl::DataTypeTraits<dataType>::type type ;

    // Compute derInput.
    if (derInput) {
      error = SubsampleBackward<deviceType,dataType>()(op,derInput,derOutput) ;
      if (error != VLE_Success) { return error ; }
    }

    // Compute derBiases.
    if (derBiases) {
      auto numOutputPixels = derOutput.getHeight() * derOutput.getWidth() ;
      type const* allOnesMemory = (type*) op.getContext().getAllOnes(deviceType, dataType, (size_t)numOutputPixels) ;

      if (allOnesMemory == NULL) {
        error = op.getContext().getLastError() ;
        goto done ;
      }

      for (Int image = 0 ; image < derInput.getCardinality() ; ++image) {
        auto derOutputOffset = (derOutput.getHeight()*derOutput.getWidth()*derOutput.getNumChannels()) * image ;
        type alpha = 1 ;
        type beta = (image > 0) ; // Avoids having to clear derOutputs first.
        error = vl::impl::blas<deviceType,dataType>::gemv
        (op.getContext(),
         't',
         numOutputPixels, derOutput.getNumChannels(),
         alpha,
         (type const*)derOutput.getMemory() + derOutputOffset, numOutputPixels,
         allOnesMemory, 1,
         beta,
         (type*)derBiases.getMemory(), 1) ;
        if (error != vl::VLE_Success) { goto done ; }
      }
    }

  done:
    return op.getContext().passError(error,signature.c_str()) ;
  }
} ;
