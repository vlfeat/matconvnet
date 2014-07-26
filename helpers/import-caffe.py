#! /usr/bin/python
# file: import-caffe.py
# brief: Caffe importer
# author: Andrea Vedaldi and Karel Lenc

import sys
import os
import argparse
import code
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
  avgim_nm, avgim_ext = os.path.splitext(args.average_image.name)
  if avgim_ext == '.binaryproto':
    blob=caffe_pb2.BlobProto()
    blob.MergeFromString(args.average_image.read())
    average_image = np.squeeze(blobproto_to_array(blob).astype('float32'),3)
    average_image = average_image.transpose([2, 3, 1, 0])
  elif avgim_ext == '.mat':
    avgim_data = sio.loadmat(args.average_image)
    average_image = avgim_data['mean_img']
    average_image = average_image.transpose([1, 0, 2])
  else:
    print 'Unsupported average image format {}'.format(avgim_ext)

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

matlab_layers = []
layers_name_param = [x.layer.name for x in net_param.layers]
layers_name_data = [x.layer.name for x in net_data.layers]
print 'Converting {} layers'.format(len(net_param.layers))

pool_methods = ['max', 'avg']
prev_out_sz = [net_param.input_dim[2],
               net_param.input_dim[3],
               net_param.input_dim[1]]

def keyboard(banner=None):
    ''' Function that mimics the matlab keyboard command '''
    # use exception trick to pick up the current frame
    try:
        raise None
    except:
        frame = sys.exc_info()[2].tb_frame.f_back
    print "# Use quit() to exit :) Happy debugging!"
    # evaluate commands in current namespace
    namespace = frame.f_globals.copy()
    namespace.update(frame.f_locals)
    try:
        code.interact(banner=banner, local=namespace)
    except SystemExit:
        return

def get_output_size(size, filter_support, pad, stride):
  return [ \
      floor((size[0] + pad[0]+pad[1] - filter_support[0]) / stride[0]) + 1, \
      floor((size[1] + pad[2]+pad[3] - filter_support[1]) / stride[1]) + 1]

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
  if layer.type == 'conv': # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mk['type'] = 'conv'
    if len(arrays) >= 1:
      mk['filters'] = arrays[0].transpose([2,3,1,0])
    else:
      mk['filters'] = np.zeros([layer.kernelsize,layer.kernelsize,
                                prev_out_sz[3],layer.num_output],dtype='float32')
    if len(arrays) >= 2:
      mk['biases'] = np.squeeze(arrays[1].transpose([2,3,1,0]), (2,3))
    else:
      mk['biases'] = np.zeros([1,layer.num_output],dtype='float32')
    if hasattr(layer, 'pad'):
      pad = float(layer.pad)
    else:
      pad = 0
    mk['pad'] = float(pad) * np.array([1.,1.,1.,1.])
    mk['stride'] = float(layer.stride) * np.array([1.,1.])
    prev_out_sz = get_output_size(prev_out_sz,
                                  mk['filters'].shape,
                                  mk['pad'],
                                  mk['stride']) + [mk['filters'].shape[3]]
  elif layer.type == 'pad': # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mk['type'] = 'pad'
    pass
  elif layer.type == 'relu': # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mk['type'] = 'relu'
  elif layer.type == 'lrn': # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mk['type'] = 'normalize'
    mk['param'] = np.array([layer.local_size, layer.k, layer.alpha, layer.beta])
  elif layer.type == 'pool': # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mk['type'] = 'pool'
    mk['pool'] = float(layer.kernelsize)*np.array([1.,1.])
    mk['method'] = pool_methods[layer.pool]
    if hasattr(layer, 'pad'):
      pad = layer.pad
    else:
      pad = 0
    mk['pad'] = float(pad)*np.array([1.,1.,1.,1.])
    # Add single pixel right/bottom padding for even sized inputs
    if prev_out_sz[0] % 2 == 0: mk['pad'][1] += 1
    if prev_out_sz[1] % 2 == 0: mk['pad'][3] += 1
    mk['stride'] = float(layer.stride)*np.array([1.,1.])
    prev_out_sz = get_output_size(prev_out_sz,
                                  mk['pool'],
                                  mk['pad'],
                                  mk['stride']) + [prev_out_sz[2]]
  elif layer.type == 'innerproduct': # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mk['type'] = 'conv'
    if len(arrays) >= 1:
      mk['filters'] = arrays[0].reshape((prev_out_sz[1], prev_out_sz[0],
                                         prev_out_sz[2], layer.num_output))
      mk['filters'].transpose([1, 0, 2, 3])
    else:
      mk['filters'] = np.zeros([prev_out_sz[1], prev_out_sz[0],
                                prev_out_sz[2], layer.num_output],dtype='float32')
    if len(arrays) >= 2:
      mk['biases'] = np.squeeze(arrays[1].transpose([2, 3, 1, 0]), (2,3))
    else:
      mk['biases'] = np.zeros([0,0],dtype='float32')
    mk['pad'] = np.array([0.,0.,0.,0.])
    mk['stride'] = np.array([1.,1.])
    prev_out_sz = [1, 1, mk['filters'].shape[3]]
  elif layer.type == 'dropout':
    # dropout ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mk['type'] = 'dropout'
    mk['rate']= float(layer.dropout_ratio)
  elif layer.type == 'softmax':
    # softmax ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mk['type'] = 'softmax'
  else:
    # anything else ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
if average_image is not None:
  mkn['averageImage']=average_image
else:
  mkn['averageImage']=np.array([0,0],dtype='float32')

# --------------------------------------------------------------------
#                                                          Save output
# --------------------------------------------------------------------

print 'Exporting to {}'.format(args.output.name)
sio.savemat(args.output, {
  'layers':np.array(matlab_layers),
  'normalization':mkn})
