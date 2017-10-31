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

