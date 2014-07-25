#! /usr/bin/python
# file: import-caffe.py
# brief: Caffe importer
# author: Andrea Vedaldi and Karel Lenc

import sys
import os
import argparse
import numpy as np
from math import floor, ceil
from numpy import array
import scipy.io as sio
import google.protobuf

# --------------------------------------------------------------------
#                                                     Helper functions
# --------------------------------------------------------------------

def blobproto_to_array(blob):
  return np.array(blob.data).reshape(
    blob.num, blob.channels, blob.height, blob.width)

# --------------------------------------------------------------------
#                                                        Parse options
# --------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Convert a Caffe CNN into a MATLAB structure.')
parser.add_argument('caffe_param',
                    type=argparse.FileType('rb'),
                    help='The Caffe CNN parameter file (ASCII .proto)')
parser.add_argument('caffe_data',
                    type=argparse.FileType('rb'),
                    help='The Caffe CNN data file (binary .proto)')
parser.add_argument('output',
                    type=argparse.FileType('w'),
                    help='Output MATLAB file')
parser.add_argument('--average-image',
                    type=argparse.FileType('rb'),
                    nargs='?',
                    help='Average image')
parser.add_argument('--synsets',
                    type=argparse.FileType('r'),
                    nargs='?',
                    help='Synset file (ASCII)')
parser.add_argument('--caffe-variant',
                    type=str,
                    nargs='?',
                    default='caffe',
                    help='Synset file (ASCII)')
args = parser.parse_args()

# --------------------------------------------------------------------
#                                                   Load average image
# --------------------------------------------------------------------

average_image = None
if args.average_image:
  print 'Loading average image'
  blob=caffe_pb2.BlobProto()
  blob.MergeFromString(args.average_image.read())
  average_image = np.squeeze(blobproto_to_array(blob) \
                               .astype('float32') \
                               .transpose([2, 3, 1, 0]), 3)

# --------------------------------------------------------------------
#                                                        Load synseths
# --------------------------------------------------------------------


# --------------------------------------------------------------------
#                                                          Load layers
# --------------------------------------------------------------------

print 'Caffe varaint set to', args.caffe_variant
if args.caffe_variant == 'vgg-caffe':
  import vgg_caffe_pb2 as caffe_pb2
elif args.caffe_variant == 'caffe':
  import caffe_pb2 as caffe_pb2

print 'Loading Caffe CNN parameters from {}'.format(args.caffe_param.name)
net_param=caffe_pb2.NetParameter()
google.protobuf.text_format.Merge(args.caffe_param.read(), net_param)

print 'Loading Caffe CNN data from {}'.format(args.caffe_data.name)
net_data=caffe_pb2.NetParameter()
net_data.MergeFromString(args.caffe_data.read())

# --------------------------------------------------------------------
#                                                       Convert layers
# --------------------------------------------------------------------

pool_methods = ['max', 'avg']

# TODO set layer sizes when blobs not provided
# TODO add command line options (netdef, netdata, schemefile)

matlab_layers = []
layers_name_param = [x.layer.name for x in net_param.layers]
layers_name_data = [x.layer.name for x in net_data.layers]
print 'Converting {} layers'.format(len(net_param.layers))

prev_out_sz = [net_param.input_dim[2],
               net_param.input_dim[3],
               net_param.input_dim[1]]

# scan all layers in net_param
for name in layers_name_param:
  index = layers_name_param.index(name)
  layer = net_param.layers[index].layer
  print 'Processing layer {} ({})'.format(index, name)
  print '  Layer input size: {} {} {}'.format(prev_out_sz[0], prev_out_sz[1], prev_out_sz[2])

  # search for a corresponding layer in net_data
  arrays = []
  if name in layers_name_data:
    index = layers_name_data.index(name)
    blobs = list(net_data.layers[index].layer.blobs)
    for b in blobs:
      arrays.append(blobproto_to_array(b).astype('float32'))
      print '  Extracted a blob of size', arrays[-1].shape

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
      mk['pad'] = float(layer.pad)
    else:
      mk['pad'] = 0.
    mk['stride'] = float(layer.stride)
    prev_out_sz = [\
      floor((prev_out_sz[0] - mk['filters'].shape[0]) / layer.stride) + 1 + 2*mk['pad'], \
      floor((prev_out_sz[1] - mk['filters'].shape[1]) / layer.stride) + 1 + 2*mk['pad'], \
      mk['filters'].shape[3]]
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
    mk['pool'] = float(layer.kernelsize)
    mk['method'] = pool_methods[layer.pool]
    pad = 0
    if hasattr(layer, 'pad'):
      mk['pad'] = float(layer.pad)*np.array([1., 1., 1., 1.])
      pad = layer.pad
    else:
      mk['pad'] = np.array([0., 0., 0., 0.])
    # Add single pixel right/bottom padding for even sized inputs
    if prev_out_sz[0] % 2 == 0:
      mk['pad'][1] += 1
    if prev_out_sz[1] % 2 == 0:
      mk['pad'][3] += 1
    mk['stride'] = float(layer.stride)
    prev_out_sz = [\
      ceil((prev_out_sz[0] - mk['pool']) / layer.stride) + 1 + 2*pad, \
      ceil((prev_out_sz[1] - mk['pool']) / layer.stride) + 1 + 2*pad, \
      prev_out_sz[2]]
  elif layer.type == 'innerproduct':
    mk['type'] = 'conv'
    if len(arrays) >= 1:
      mk['filters'] = arrays[0].reshape((prev_out_sz[1], prev_out_sz[0], prev_out_sz[2], -1))
      mk['filters'].transpose([1, 0, 2, 3])
    else:
      mk['filters'] = np.zeros([0,0],dtype='float32')
    if len(arrays) >= 2:
      mk['biases'] = np.squeeze(arrays[1].transpose([2, 3, 1, 0]), (2,3))
    else:
      mk['biases'] = np.zeros([0,0],dtype='float32')
    mk['pad'] = 0.
    mk['stride'] = 1.
    prev_out_sz = [ 1, 1, mk['filters'].shape[3]];
  elif layer.type == 'dropout':
    mk['type'] = 'dropout'
    mk['rate']= float(layer.dropout_ratio)
  elif layer.type == 'softmax':
    mk['type'] = 'softmax'
  else:
    mk['type'] = layer.type
    print 'Warning: unknown layer type ', layer.type
  matlab_layers.append(mk)

mkn = {}
if len(net_param.input_dim) > 0:
  mkn['imageSize']=np.array([ \
      net_param.input_dim[2], \
        net_param.input_dim[3], \
        net_param.input_dim[1]],dtype='float64')
else:
  mkn['imageSize']=np.array([0,0],dtype='float32')
if average_image:
  mkn['averageImage']=average_image
else:
  mkn['averageImage']=np.array([0,0],dtype='float32')

# --------------------------------------------------------------------
#                                                          Save output
# --------------------------------------------------------------------

print 'Exporting to {}'.format(args.output)
scipy.io.savemat(args.output, {
  'layers':np.array(matlab_layers),
  'normalization':mkn})
