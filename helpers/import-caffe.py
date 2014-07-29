#! /usr/bin/python
# file: import-caffe.py
# brief: Caffe importer
# author: Andrea Vedaldi and Karel Lenc

import sys
import os
import argparse
import code
import re
import numpy as np
from math import floor, ceil
from numpy import array
import scipy
import scipy.io
import scipy.misc
import google.protobuf

# --------------------------------------------------------------------
#                                                     Helper functions
# --------------------------------------------------------------------

def blobproto_to_array(blob):
  return np.array(blob.data,dtype='float32').reshape(
    blob.num, blob.channels, blob.height, blob.width).transpose()

layers_type = {}
layers_type[0]  = 'none'
layers_type[1]  = 'accuracy'
layers_type[2]  = 'bnll'
layers_type[3]  = 'concat'
layers_type[4]  = 'conv'
layers_type[5]  = 'data'
layers_type[6]  = 'dropout'
layers_type[7]  = 'euclidean_loss'
layers_type[25] = 'eltwise_product'
layers_type[8]  = 'flatten'
layers_type[9]  = 'hdf5_data'
layers_type[10] = 'hdf5_output'
layers_type[28] = 'hinge_loss'
layers_type[11] = 'im2col'
layers_type[12] = 'image_data'
layers_type[13] = 'infogain_loss'
layers_type[14] = 'inner_product'
layers_type[15] = 'lrn'
layers_type[29] = 'memory_data'
layers_type[16] = 'multinomial_logistic_loss'
layers_type[17] = 'pool'
layers_type[26] = 'power'
layers_type[18] = 'relu'
layers_type[19] = 'sigmoid'
layers_type[27] = 'sigmoid_cross_entropy_loss'
layers_type[20] = 'softmax'
layers_type[21] = 'softmax_loss'
layers_type[22] = 'split'
layers_type[23] = 'tanh'
layers_type[24] = 'window_data'

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
                    help='Variant of Caffe software (use ? to get a list)')
parser.add_argument('--transpose',
                    dest='transpose',
                    action='store_true',
                    help='Transpose CNN in a sane MATLAB format')
parser.add_argument('--no-transpose',
                    dest='transpose',
                    action='store_false',
                    help='Do not transpose CNN')
parser.set_defaults(transpose=True)
args = parser.parse_args()

print 'Caffe varaint set to', args.caffe_variant
if args.caffe_variant == 'vgg-caffe':
  import proto.vgg_caffe_pb2 as caffe_pb2
elif args.caffe_variant == 'caffe-old':
  import proto.caffe_old_pb2 as caffe_pb2
elif args.caffe_variant == 'caffe':
  import proto.caffe_pb2 as caffe_pb2
elif args.caffe_variant == '?':
  print 'Supported variants: caffe, cafe-old, vgg-caffe'
  sys.exit(0)
else:
  print 'Uknown Caffe variant', args.caffe_variant
  sys.exit(1)

# --------------------------------------------------------------------
#                                                     Helper functions
# --------------------------------------------------------------------

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
      (size[0] + pad[0]+pad[1] - filter_support[0]) / stride[0] + 1, \
      (size[1] + pad[2]+pad[3] - filter_support[1]) / stride[1] + 1]

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
    average_image = blobproto_to_array(blob).astype('float32')
    average_image = np.squeeze(average_image,3)
    if args.transpose and average_image is not None:
      average_image = average_image.transpose([1,0,2])
      average_image = average_image[:,:,: : -1] # to RGB
  elif avgim_ext == '.mat':
    avgim_data = scipy.io.loadmat(args.average_image)
    average_image = avgim_data['mean_img']
  else:
    print 'Unsupported average image format {}'.format(avgim_ext)

# --------------------------------------------------------------------
#                                                        Load synseths
# --------------------------------------------------------------------

synsets_wnid=None
synsets_name=None
if args.synsets:
  print 'Loading synsets'
  r=re.compile('(?P<wnid>n[0-9]{8}?) (?P<name>.*)')
  synsets_wnid=[]
  synsets_name=[]
  for line in args.synsets:
    match = r.match(line)
    synsets_wnid.append(match.group('wnid'))
    synsets_name.append(match.group('name'))

# --------------------------------------------------------------------
#                                                          Load layers
# --------------------------------------------------------------------

print 'Loading Caffe CNN parameters from {}'.format(args.caffe_param.name)
net_param=caffe_pb2.NetParameter()
google.protobuf.text_format.Merge(args.caffe_param.read(), net_param)

print 'Loading Caffe CNN data from {}'.format(args.caffe_data.name)
net_data=caffe_pb2.NetParameter()
net_data.MergeFromString(args.caffe_data.read())

# --------------------------------------------------------------------
#                                                       Convert layers
# --------------------------------------------------------------------

if args.caffe_variant in ['vgg-caffe', 'caffe-old']:
  layers_name_param = [x.layer.name for x in net_param.layers]
  layers_name_data = [x.layer.name for x in net_data.layers]
else:
  layers_name_param = [x.name for x in net_param.layers]
  layers_name_data = [x.name for x in net_data.layers]

pool_methods = ['max', 'avg']
layer_input_size = [net_param.input_dim[2],
                    net_param.input_dim[3],
                    net_param.input_dim[1]]

print 'Converting {} layers'.format(len(net_param.layers))
print layers_name_param
print layers_name_data

# scan all layers in net_param
matlab_layers = []
for name in layers_name_param:
  index = layers_name_param.index(name)
  layer = net_param.layers[index]
  if args.caffe_variant in ['vgg-caffe', 'caffe-old']: layer=layer.layer
  ltype = layer.type
  if not isinstance(ltype, basestring): ltype = layers_type[ltype]

  print 'Processing layer {} ({}, {})'.format(index, name, ltype)
  print '  Layer input size: {} {} {}'.format(layer_input_size[0],
                                              layer_input_size[1],
                                              layer_input_size[2])

  # search for a corresponding layer in net_data
  arrays = []
  param = layer
  support = [1,1]
  pad = [0,0,0,0]
  stride = [1,1]
  num_output_channels = layer_input_size[2]

  if name in layers_name_data:
    index = layers_name_data.index(name)
    if args.caffe_variant in ['caffe']:
      layer_data = net_data.layers[index]
    else:
      layer_data = net_data.layers[index].layer
    blobs = list(layer_data.blobs)
    for b in blobs:
      arrays.append(blobproto_to_array(b).astype('float32'))
      print '  Extracted a blob of size', arrays[-1].shape

  mk = {'name': layer.name}
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if ltype == 'conv':
    mk['type'] = 'conv'
    if hasattr(layer, 'convolution_param'): param = layer.convolution_param
    if hasattr(layer, 'kernelsize'): support = [param.kernelsize]*2
    else: support = [param.kernel_size]*2
    pad = [param.pad]*4
    stride = [param.stride]*2
    num_output_channels = param.num_output
    if len(arrays) >= 1:
      mk['filters'] = arrays[0]
    else:
      mk['filters'] = np.zeros(support + [layer_input_size[2], num_output_channels],
                               dtype='float32')
    if len(arrays) >= 2:
      mk['biases'] = np.squeeze(arrays[1], (2,3))
    else:
      mk['biases'] = np.zeros([1,num_output_channels],dtype='float32')
    mk['pad'] = pad
    mk['stride'] = stride
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype == 'relu':
    mk['type'] = 'relu'
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype == 'lrn':
    mk['type'] = 'normalize'
    if hasattr(layer, 'lrn_param'): param = layer.lrn_param
    local_size = float(param.local_size)
    alpha = float(param.alpha)
    beta = float(param.beta)
    kappa = 1.
    if hasattr(param, 'k'): kappa = param.k
    mk['param'] = np.array([local_size, kappa, alpha/local_size, beta])
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype == 'pool':
    mk['type'] = 'pool'
    if hasattr(layer, 'pooling_param'): param = layer.pooling_param
    if hasattr(layer, 'kernelsize'): support = [param.kernelsize]*2
    else: support = [param.kernel_size]*2
    pad = [param.pad]*4
    stride = [param.stride]*2
    if layer_input_size[0] % 2 == 0: pad[1] += 1
    if layer_input_size[1] % 2 == 0: pad[3] += 1
    mk['pool'] = support
    mk['method'] = pool_methods[param.pool]
    mk['pad'] = pad
    mk['stride'] = stride
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype == 'innerproduct' or ltype == 'inner_product':
    mk['type'] = 'conv'
    if hasattr(layer, 'inner_product_param'): param = layer.inner_product_param
    support = [layer_input_size[0], layer_input_size[1]]
    pad = [0]*4
    stride = [1]*2
    num_output_channels = param.num_output
    if len(arrays) >= 1:
      mk['filters'] = arrays[0].reshape(
        layer_input_size[0],
        layer_input_size[1],
        layer_input_size[2],
        num_output_channels,
        order='F')
    else:
      mk['filters'] = np.zeros([layer_input_size[0],
                                layer_input_size[1],
                                layer_input_size[2],
                                num_output_channels],dtype='float32')
    if len(arrays) >= 2:
      mk['biases'] = np.squeeze(arrays[1], (2,3))
    else:
      mk['biases'] = np.zeros([1,num_output_channels],dtype='float32')
    mk['pad'] = pad
    mk['stride'] = stride
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype == 'dropout':
    mk['type'] = 'dropout'
    if hasattr(layer, 'dropout_param'): param = layer.dropout_param
    mk['rate']= float(param.dropout_ratio)
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype == 'softmax':
    mk['type'] = 'softmax'
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  else:
    mk['type'] = ltype
    print 'Warning: unknown layer type', ltype
  print '  Support:',support
  print '  Pad:',pad
  print '  Stride:',stride
  for f in ['pad', 'stride', 'pool']:
    if f in mk: mk[f] = [float(i) for i in mk[f]]
  layer_input_size = get_output_size(layer_input_size,
                                     support, pad, stride) + [num_output_channels]
  matlab_layers.append(mk)


# --------------------------------------------------------------------
#                                                Reshape and transpose
# --------------------------------------------------------------------

first_conv_layer = True
for i in range(0,len(matlab_layers)):
  for f in ['pad', 'stride', 'pool', 'param']:
    if f in matlab_layers[i]:
      matlab_layers[i][f] = np.array(matlab_layers[i][f],dtype=float).reshape([1,-1])
  if matlab_layers[i]['type'] == 'conv':
    matlab_layers[i]['biases'] = matlab_layers[i]['biases'].reshape(1,-1) # row
    if args.transpose:
      matlab_layers[i]['filters'] = matlab_layers[i]['filters'].transpose([1,0,2,3])
      if first_conv_layer:
        if not args.caffe_variant in ['vgg-caffe']:
          matlab_layers[i]['filters'] = matlab_layers[i]['filters'][:,:,: : -1,:] # to RGB
        first_conv_layer = False

# --------------------------------------------------------------------
#                                                        Normalization
# --------------------------------------------------------------------

mkn = {}
if len(net_param.input_dim) > 0:
  mkn['imageSize']=np.array([ \
      net_param.input_dim[2], \
        net_param.input_dim[3], \
        net_param.input_dim[1]],dtype='float64')
else:
  mkn['imageSize']=np.array([0,0],dtype='float32')
if average_image is not None:
  mkn['averageImage']=scipy.misc.imresize(average_image,
                                          (mkn['imageSize'][0], mkn['imageSize'][1]),
                                          interp='bilinear')
else:
  mkn['averageImage']=np.array([0,0],dtype='float32')

# --------------------------------------------------------------------
#                                                          Save output
# --------------------------------------------------------------------

print 'Exporting to {}'.format(args.output.name)

mnet = {
  'layers': np.array(matlab_layers).reshape(1,-1),
  'normalization': mkn}
if synsets_wnid: mnet['wnid'] = np.array(synsets_wnid, dtype=np.object).reshape(1,-1)
if synsets_name: mnet['classes'] = np.array(synsets_name, dtype=np.object).reshape(1,-1)

scipy.io.savemat(args.output, mnet)

