#! /usr/bin/python

#import caffe_pb2
import vgg_caffe_pb2 as caffe_pb2
import google.protobuf
import numpy as np
import scipy.io as sio
from numpy import array

def blobproto_to_array(blob, return_diff=False):
  """Convert a blob proto to an array. In default, we will just return the data,
  unless return_diff is True, in which case we will return the diff.
  """
  if return_diff:
    return np.array(blob.diff).reshape(
        blob.num, blob.channels, blob.height, blob.width)
  else:
    return np.array(blob.data).reshape(
        blob.num, blob.channels, blob.height, blob.width)

param_file='../data/vgg/CNN_F/param.prototxt'
model_file='../data/vgg/CNN_F/model'

# load network parameters
net=caffe_pb2.NetParameter()
with open(param_file, 'r') as f:
  google.protobuf.text_format.Merge(f.read(), net)
net_bin=caffe_pb2.NetParameter()
with open(model_file, 'rb') as f:
  net_bin.MergeFromString(f.read())

matlabLayers = []
layers = [x.layer.name for x in net.layers]
layers_bin = [x.layer.name for x in net_bin.layers]
print layers
for name in layers:
    index = layers.index(name)
    layer = net.layers[index].layer

    arrays = []
    if name in layers_bin:
      index = layers_bin.index(name)
      blobs = list(net_bin.layers[index].layer.blobs)
      for b in blobs:
        num = b.num
        channels = b.channels
        height = b.height
        width = b.width
        blobIndex = blobs.index(b)
        print "extracting blob %s (%d x %d x %d x %d)" \
            % (name, height, width, channels, num)
        arrays.append(blobproto_to_array(b).astype('float32'))

    mk = {'name': layer.name}
    if layer.type == 'conv':
      mk['type'] = 'conv'
      if len(arrays) >= 1:
        mk['filters'] = arrays[0].transpose([2, 3, 1, 0])
      else:
        mk['filters'] = np.zeros([0,0],dtype='float32')
      if len(arrays) >= 2:
        mk['biases'] = np.squeeze(arrays[1].transpose([2, 3, 1, 0]), (2,3))
      else:
        mk['biases'] = np.zeros([0,0],dtype='float32')
      if hasattr(layer, 'pad'):
        mk['pad'] = layer.pad
      else:
        mk['pad'] = 0
      mk['stride'] = layer.stride
    elif layer.type == 'pad':
      mk['type'] = 'pad'
      pass
    elif layer.type == 'relu':
      mk['type'] = 'relu'
    elif layer.type == 'lrn':
      mk['type'] = 'normalize'
      mk['param'] = np.array([layer.local_size, layer.k, layer.alpha, layer.beta])
    elif layer.type == 'pool':
      mk['type'] = 'pool'
      mk['pool'] = layer.kernelsize
      mk['method'] = layer.pool
      if hasattr(layer, 'pad'):
        mk['pad'] = layer.pad
      else:
        mk['pad'] = 0
      mk['stride'] = layer.stride
    elif layer.type == 'innerproduct':
      mk['type'] = 'conv'
      if len(arrays) >= 1:
        mk['filters'] = arrays[0].transpose([0, 1, 3, 2])
      else:
        mk['filters'] = np.zeros([0,0],dtype='float32')
      if len(arrays) >= 2:
        mk['biases'] = np.squeeze(arrays[1].transpose([2, 3, 1, 0]), (2,3))
      else:
        mk['biases'] = np.zeros([0,0],dtype='float32')
      mk['pad'] = 0
      mk['stride'] = 1
    elif layer.type == 'dropout':
      mk['type'] = 'dropout'
      mk['rate']= layer.dropout_ratio
    elif layer.type == 'softmax':
      mk['type'] = 'softmax'
    else:
      mk['type'] = layer.type
      print 'Warning: unknown layer type ', layer.type

    matlabLayers.append(mk)

mkNormalization = {}
if len(net.input_dim) > 0:
  mkNormalization['imageSize']=np.array([\
      net.input_dim[2], \
        net.input_dim[3], \
        net.input_dim[1]],dtype='float64')
else:
  mkNormalization['imageSize']=np.array([0,0],dtype='float32')
mkNormalization['averageImage']=np.array([0,0],dtype='float32')

sio.savemat('test.mat', {\
'layers':np.array(matlabLayers),\
'normalization': mkNormalization})
