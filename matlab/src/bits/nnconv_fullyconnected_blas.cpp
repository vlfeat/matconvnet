// -------------------------------------------------------------------
//                                                             Forward
// -------------------------------------------------------------------

template<DeviceType deviceType, DataType dataType>
struct FullyConnectedForward
{
  vl::ErrorCode operator()
  (Convolution const &op,
   Tensor &output, double outputMult,
   Tensor const& input, double inputMult,
   Tensor const& filter)
  {
    static const std::string signature = std::string("FullyConnectedForward[BLAS,")
    + DeviceTypeTraits<deviceType>::name + "," + DataTypeTraits<dataType>::name + "]" ;

    VLLOG(op,1) << signature ;

    vl::ErrorCode error = VLE_Success ;
    typedef typename vl::DataTypeTraits<dataType>::type type ;
    type alpha = (type)inputMult ;
    type beta = (type)outputMult ;

    if (filter) {
      auto filterVolume = filter.getHeight() * filter.getWidth() * filter.getNumChannels() ;
      if (input.getCardinality() == 1) {
        /* one image in the stack */
        error = vl::impl::blas<deviceType,dataType>::gemv
        (op.getContext(),
         't',
         filterVolume, filter.getCardinality(),
         alpha,
         (type const*)filter.getMemory(), filterVolume,
         (type const*)input.getMemory(), 1,
         beta,
         (type*)output.getMemory(), 1) ;
        if (error != vl::VLE_Success) { goto done ; }
      } else {
        /* multiple images in the stack */
        error = vl::impl::blas<deviceType,dataType>::gemm
        (op.getContext(),
         't', 'n',
         filter.getCardinality(),
         input.getCardinality(),
         filterVolume,
         alpha,
         (type const*)filter.getMemory(), filterVolume,
         (type const*)input.getMemory(), filterVolume,
         beta,
         (type*)output.getMemory(), filter.getCardinality()) ;
        if (error != vl::VLE_Success) { goto done ; }
      }
    } else {
      error = vl::impl::operations<deviceType,type>::copy
      ((type*)output.getMemory(),
       (type const*)input.getMemory(),
       (size_t)input.getNumElements()) ;
    }

  done:
    return op.getContext().passError(error,signature.c_str()) ;
  }
};

// -------------------------------------------------------------------
//                                                            Backward
// -------------------------------------------------------------------

template<DeviceType deviceType, DataType dataType>
struct FullyConnectedBackward
{
  vl::ErrorCode operator()
  (Convolution const &op,
   vl::Tensor &derInput,
   vl::Tensor &derFilter,
   vl::Tensor const &input,
   vl::Tensor const &filter,
   vl::Tensor const &derOutput)
  {
    static const std::string signature = std::string("FullyConnectedBackward[BLAS,")
    + DeviceTypeTraits<deviceType>::name + "," + DataTypeTraits<dataType>::name + "]" ;

    VLLOG(op,1) << signature ;

    vl::ErrorCode error = VLE_Success ;
    typedef typename vl::DataTypeTraits<dataType>::type type ;
    type alpha = 1 ;
    type beta = 0 ;

    if (filter) {
      auto filterVolume = filter.getHeight() * filter.getWidth() * filter.getNumChannels() ;

      if (derFilter) {
        error = vl::impl::blas<deviceType,dataType>::gemm
        (op.getContext(),
         'n', 't',
         filterVolume,
         filter.getCardinality(),
         input.getCardinality(),
         alpha,
         (type*)input.getMemory(), filterVolume,
         (type*)derOutput.getMemory(), filter.getCardinality(),
         beta,
         (type*)derFilter.getMemory(), filterVolume) ;
        if (error != vl::VLE_Success) { goto done ; }
      }

      if (derInput) {
        error = vl::impl::blas<deviceType,dataType>::gemm
        (op.getContext(),
         'n', 'n',
         filterVolume,
         input.getCardinality(),
         filter.getCardinality(),
         alpha,
         (type*)filter.getMemory(), filterVolume,
         (type*)derOutput.getMemory(), filter.getCardinality(),
         beta,
         (type*)derInput.getMemory(), filterVolume) ;
        if (error != vl::VLE_Success) { goto done ; }
      }
    } else {
      vl::impl::operations<deviceType,type>::copy
      ((type*)derInput.getMemory(),
       (type const*)derOutput.getMemory(),
       (size_t)derOutput.getNumElements()) ;
    }
 
  done:
    return op.getContext().passError(error,signature.c_str()) ;
  }
};
