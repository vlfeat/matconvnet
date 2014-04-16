function im = caffe_denormalize(net, im)

% premute as needed
inverseChannelPerm(net.normalization.channelPerm) = 1:3; 
inverseDimensionPerm(net.normalization.dimensionPerm) = 1:3 ;
im = permute(im(:,:,inverseChannelPerm), inverseDimensionPerm) ;

% take mean in
im = im + net.normalization.meanImage ;

% convert to MATLAB convention for float image
im = im/255 ;


