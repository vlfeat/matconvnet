// -------------------------------------------------------------------
/// MARK: - Convolution
// -------------------------------------------------------------------

/*
 One image at a time is processed.

 Filters are (optionally) divided in to groups, one for each group of dimensions.


                 patchVolume                  numFilters
                 +-------------------------+   +-----------------------+

                 filterVolume              numFiltersPerGroup
                 +------------+------------+   +-----------+-----------+      +--------+--------+
                 |            |            |   |           |           |      |        |        |
                 |            |            |   |  filter   |           |      |        |        |
                 |            |            |   |  group 1  |     0     |  =   |        |        |
                 |            |            |   |           |           |      |        |        |
                 |            |            |   |           |           |      |        |        |
                 |            |            |   +-----------------------+      |        |        |
 numOutputPixels |   grp. 1   |   grp. 2   |   |           |           |      |        |        |
                 |            |            |   |           |  filter   |      |        |        |
                 |            |            |   |     0     |  group 2  |      |        |        |
                 |            |            |   |           |           |      |        |        |
                 |            |            |   |           |           |      |        |        |
                 |            |            |   +-----------+-----------+      |        |        |
                 |            |            |                                  |        |        |
                 |            |            |   filters                        |        |        |
                 |            |            |                                  |        |        |
                 +------------+------------+                                  +--------+--------+

                 temp                                                         output

 */


template<DeviceType deviceType, DataType dataType>
struct ConvolutionForward
{
  vl::ErrorCode operator()
  (Convolution &op,
   Tensor output, double outputMult,
   Tensor const& input, double inputMult,
   Tensor const& filter)
  {
    assert(output) ;
    assert(input) ;
    assert(filter) ;
    assert(input.getNumDimensions() <= 4) ; // Todo: generalize.

    static const std::string signature = std::string("ConvolutionForward[BLAS,")
    + DeviceTypeTraits<deviceType>::name + "," + DataTypeTraits<dataType>::name + "]" ;

    VLLOG(op,1) << signature ;

    vl::ErrorCode error = VLE_Success ;
    typedef typename vl::DataTypeTraits<dataType>::type type ;

    Int numGroups = input.getNumChannels() / filter.getNumChannels() ;
    Int numFiltersPerGroup = filter.getCardinality() / numGroups ;
    Int numOutputPixels = output.getHeight() * output.getWidth() ;
    Int filterVolume = filter.getHeight() * filter.getWidth() * filter.getNumChannels() ;
    Int tempVolume = numOutputPixels * filterVolume * numGroups ;

    type* tempMemory = (type*) op.getContext().getWorkspace
    (deviceType, as_unsigned(tempVolume) * sizeof(type)) ;

    if (tempMemory == NULL) {
      error = op.getContext().getLastError() ;
      goto done ;
    }

    for (Int image = 0 ; image < input.getCardinality() ; ++image) {

      auto dataOffset = (input.getHeight()*input.getWidth()*input.getNumChannels()) * image ;
      auto outputOffset = (output.getHeight()*output.getWidth()*output.getNumChannels()) * image ;

      error = vl::impl::im2row<deviceType,type>::forward
      (op.getContext(),
       tempMemory,
       (type*)input.getMemory() + dataOffset,
       input.getHeight(), input.getWidth(), input.getNumChannels(),
       filter.getHeight(), filter.getWidth(),
       op.getStride(0),
       op.getStride(1),
       op.getPadding(0),
       op.getPadding(1),
       op.getPadding(2),
       op.getPadding(3),
       op.getDilation(0),
       op.getDilation(1)) ;
      if (error != vl::VLE_Success) { goto done ; }

      for (Int g = 0 ; g < numGroups ; ++ g) {
        Int filterGrpOffset = filterVolume * numFiltersPerGroup * g ;
        Int tempGrpOffset = numOutputPixels * filterVolume * g ;
        Int outputGrpOffset = numOutputPixels * numFiltersPerGroup * g  ;
        auto alpha = static_cast<type>(inputMult) ;
        auto beta = static_cast<type>(outputMult) ;
        error = vl::impl::blas<deviceType,dataType>::gemm
        (op.getContext(),
         'n', 'n',
         numOutputPixels, numFiltersPerGroup, filterVolume,
         alpha,
         tempMemory + tempGrpOffset, numOutputPixels,
         (type*)filter.getMemory() + filterGrpOffset, filterVolume,
         beta,
         (type*)output.getMemory() + outputOffset + outputGrpOffset, numOutputPixels) ;
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

template<DeviceType deviceType, DataType dataType>
struct ConvolutionBackward
{
  vl::ErrorCode operator()
  (Convolution &op,
   Tensor &derInput,
   Tensor &derFilter,
   Tensor const &input,
   Tensor const &filter,
   Tensor const &derOutput)
  {
    static const std::string signature = std::string("ConvolutionBackward[BLAS,")
    + DeviceTypeTraits<deviceType>::name + "," + DataTypeTraits<dataType>::name + "]" ;

    VLLOG(op,1) << signature ;

    vl::ErrorCode error = VLE_Success ;
    typedef typename vl::DataTypeTraits<dataType>::type type ;

    Int numGroups = 0 ;
    Int numFiltersPerGroup = 0 ;
    Int filterVolume = 0 ;
    Int tempVolume = 0 ;
    type* tempMemory = NULL ;

    // for all derivatives
    assert(derOutput) ;
    Int numOutputPixels = derOutput.getHeight() * derOutput.getWidth() ;

    if (derInput) {
      // for derivative w.r.t. data
      assert(filter) ;
      numGroups = derInput.getNumChannels() / filter.getNumChannels() ;
      filterVolume = filter.getHeight() * filter.getWidth() * filter.getNumChannels() ;
    }
    else if (derFilter) {
      // for derivative w.r.t. filter
      assert(input) ;
      numGroups = input.getNumChannels() / derFilter.getNumChannels() ;
      filterVolume = derFilter.getHeight() * derFilter.getWidth() * derFilter.getNumChannels() ;
    }
    numFiltersPerGroup = derOutput.getNumChannels() / numGroups ;

    // get scratch space
    tempVolume = numOutputPixels * filterVolume * numGroups ;
    if (tempVolume) {
      tempMemory = (type*) op.getContext().getWorkspace(deviceType, as_unsigned(tempVolume) * sizeof(type)) ;
      if (tempMemory == NULL) {
        error = op.getContext().getLastError() ;
        goto done ;
      }
    }

    for (Int image = 0 ; image < derOutput.getCardinality() ; ++image) {

      Int derOutputOffset = (derOutput.getHeight()*derOutput.getWidth()*derOutput.getNumChannels()) * image ;

      /* compute derInpu dz/dx */
      if (derInput) {
        // has derInpu, derOutput, filter
        Int derInpuOffset = (derInput.getHeight()*derInput.getWidth()*derInput.getNumChannels()) * image ;
        for (Int g = 0 ; g < numGroups ; ++ g) {
          Int filterGrpOffset = filterVolume * numFiltersPerGroup * g ;
          Int tempGrpOffset = numOutputPixels * filterVolume * g ;
          Int derOutputGrpOffset = numOutputPixels * numFiltersPerGroup * g  ;
          type alpha = 1 ;
          type beta = 0 ;
          error = vl::impl::blas<deviceType,dataType>::gemm
          (op.getContext(),
           'n', 't',
           numOutputPixels, filterVolume, numFiltersPerGroup,
           alpha,
           (type*)derOutput.getMemory() + derOutputOffset + derOutputGrpOffset, numOutputPixels,
           (type*)filter.getMemory() + filterGrpOffset, filterVolume,
           beta,
           tempMemory + tempGrpOffset, numOutputPixels) ;
          if (error != vl::VLE_Success) { return error ; }
        }
        error = vl::impl::im2row<deviceType,type>::backward
        (op.getContext(),
         (type*)derInput.getMemory() + derInpuOffset,
         tempMemory,
         derInput.getHeight(), derInput.getWidth(), derInput.getNumChannels(),
         filter.getHeight(), filter.getWidth(),
         op.getStride(0),
         op.getStride(1),
         op.getPadding(0),
         op.getPadding(1),
         op.getPadding(2),
         op.getPadding(3),
         op.getDilation(0),
         op.getDilation(1)) ;
        if (error != vl::VLE_Success) { return error ; }
      }

      /* compute derFilter dz/dF */
      if (derFilter) {
        // has derFilter, derOutput, data
        Int dataOffset = (input.getHeight()*input.getWidth()*input.getNumChannels()) * image ;
        error = vl::impl::im2row<deviceType,type>::forward
        (op.getContext(),
         (type*)tempMemory,
         (type*)input.getMemory() + dataOffset,
         input.getHeight(), input.getWidth(), input.getNumChannels(),
         derFilter.getHeight(), derFilter.getWidth(),
         op.getStride(0),
         op.getStride(1),
         op.getPadding(0),
         op.getPadding(1),
         op.getPadding(2),
         op.getPadding(3),
         op.getDilation(0),
         op.getDilation(1)) ;
        if (error != vl::VLE_Success) { return error ; }
        for (Int g = 0 ; g < numGroups ; ++ g) {
          Int filterGrpOffset = filterVolume * numFiltersPerGroup * g ;
          Int tempGrpOffset = numOutputPixels * filterVolume * g ;
          Int derOutputGrpOffset = numOutputPixels * numFiltersPerGroup * g  ;
          /* dzdF = temp' * dzdY */
          type alpha = 1 ;
          type beta = (image > 0) ; /* this saves init. the output array with 0 */
          error = vl::impl::blas<deviceType,dataType>::gemm
          (op.getContext(),
           't', 'n',
           filterVolume, numFiltersPerGroup, numOutputPixels,
           alpha,
           tempMemory + tempGrpOffset, numOutputPixels,
           (type*)derOutput.getMemory() + derOutputOffset + derOutputGrpOffset, numOutputPixels,
           beta,
           (type*)derFilter.getMemory() + filterGrpOffset, filterVolume) ;
          if (error != vl::VLE_Success) { return error ; }
        }
      }
    }
  done:
    return op.getContext().passError(error, signature.c_str()) ;
  }
} ;

// -------------------------------------------------------------------
/// MARK: - Convolution transpose
// -------------------------------------------------------------------

template<DeviceType deviceType, DataType dataType>
struct ConvolutionTransposeForward
{
  vl::ErrorCode operator()
  (ConvolutionTranspose &op,
   vl::Tensor &output,
   vl::Tensor const &input,
   vl::Tensor const &filter,
   vl::Tensor const &bias)
  {
    static const std::string signature = std::string("ConvolutionTransposeForward[BLAS,")
    + DeviceTypeTraits<deviceType>::name + "," + DataTypeTraits<dataType>::name + "]" ;

    VLLOG(op,1) << signature ;

    auto logLevel = op.getContext().getLogLevel() ;
    op.getContext().setLogLevel(0) ;

    vl::ErrorCode error = VLE_Success ;
    Int dataOffset = input.getHeight()*input.getWidth()*input.getNumChannels() ;
    Int outputOffset = output.getHeight()*output.getWidth()*output.getNumChannels() ;

    // we need to process this down per image as nnconv_backward would otherwise
    // accumulate everything into a single feature field in the output
    for (Int image = 0 ; image < input.getCardinality() ; ++image) {
      Tensor inputSlice(input) ;
      Tensor outputSlice(output) ;

      switch (input.getDataType()) {
        case VLDT_Float:
          inputSlice.setMemory((float*)input.getMemory() + dataOffset * image) ;
          outputSlice.setMemory((float*)output.getMemory() + outputOffset * image) ;
          break ;
        case VLDT_Double:
          inputSlice.setMemory((double*)input.getMemory() + dataOffset * image) ;
          outputSlice.setMemory((double*)output.getMemory() + outputOffset * image) ;
          break ;
        default:
          assert(false) ;
      }
      inputSlice.setSize(1) ;
      outputSlice.setSize(1) ;

      Convolution opc(op.getContext(),
                      op.getUpsampling(0), op.getUpsampling(1),
                      op.getCropping(0),
                      op.getCropping(1),
                      op.getCropping(2),
                      op.getCropping(3),
                      1, 1) ;
      Tensor null ;
      Tensor pseudoInput(outputSlice) ;
      pseudoInput.setMemory(NULL) ;
      error = opc.backward(outputSlice, null, null,
                           pseudoInput, filter, inputSlice) ;
      if (error != VLE_Success) { goto done ; }
    }
    if (bias) {
      error = vl::nn::Bias(op.getContext()).forward(output,1.0,Tensor(),0,bias,1.0);
    }
  done:
    op.getContext().setLogLevel(logLevel) ;
    return op.getContext().passError(error,signature.c_str()) ;
  }
} ;

// -------------------------------------------------------------------
//                                      Convolution Transpose Backward
// -------------------------------------------------------------------

template<DeviceType deviceType, DataType dataType>
struct ConvolutionTransposeBackward
{
  vl::ErrorCode operator()
  (ConvolutionTranspose &op,
   vl::Tensor &derInput,
   vl::Tensor &derFilter,
   vl::Tensor &derBias,
   vl::Tensor const &input,
   vl::Tensor const &filter,
   vl::Tensor const &derOutput)
  {
    static const std::string signature = std::string("ConvolutionTransposeBackward[BLAS,")
    + DeviceTypeTraits<deviceType>::name + "," + DataTypeTraits<dataType>::name + "]" ;

    VLLOG(op,1) << signature ;

    auto logLevel = op.getContext().getLogLevel() ;
    op.getContext().setLogLevel(0) ;

    vl::ErrorCode error = vl::VLE_Success ;
    Convolution opc(op.getContext(),
                    op.getUpsampling(0),
                    op.getUpsampling(1),
                    op.getCropping(0),
                    op.getCropping(1),
                    op.getCropping(2),
                    op.getCropping(3),
                    1, 1) ;
    Tensor null ;

    if (derInput) {
      error = opc.forward(derInput, 0,
                          derOutput, 1,
                          filter, null) ;
      if (error != VLE_Success) { goto done ; }
    }

    if (derFilter) {
      error = opc.backward(null, derFilter, null,
                           derOutput,
                           filter, // only used for its shape
                           input) ;
      if (error != VLE_Success) { goto done ; }
    }

    if (derBias) {
      Tensor null ;
      error = vl::nn::Bias(op.getContext()).backward(null,0,derBias,0,0,1,derOutput) ;
    }
  done:
    op.getContext().setLogLevel(logLevel) ;
    return op.getContext().passError(error,signature.c_str()) ;
  }
} ;

