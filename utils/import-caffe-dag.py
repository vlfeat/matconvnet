#! /usr/bin/python
# file: import-caffe-dag.py
# brief: Caffe importer for DagNN
# author: Karel Lenc and Andrea Vedaldi

# Requires Google Protobuf for Python and SciPy

import sys
import os
import argparse
import code
import re
import numpy as np
from math import floor, ceil
import numpy
from numpy import array
import scipy
import scipy.io
import scipy.misc
import google.protobuf
from ast import literal_eval as make_tuple

# --------------------------------------------------------------------
#                                                     Helper functions
# --------------------------------------------------------------------

def find(seq, name):
  for item in seq:
    if item.name == name:
      return item
  return None

def blobproto_to_array(blob):
  return np.array(blob.data,dtype='float32').reshape(
    blob.num, blob.channels, blob.height, blob.width).transpose()

def dict_to_struct_array(d):
  if not d:
    return np.zeros((0,))
  dt=[(x,object) for x in d.keys()]
  y = np.empty((1,),dtype=dt)
  for x in d.keys():
    y[x][0] = d[x]
  return y

def versiontuple(version):
  return tuple(map(int, (version.split("."))))

min_numpy_version = "1.7.0"
if versiontuple(numpy.version.version) < versiontuple(min_numpy_version):
  print 'Unsupported numpy version ({}), must be >= {}'.format(numpy.version.version,
    min_numpy_version)
  sys.exit(0)

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
parser.add_argument('caffe_params',
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
parser.add_argument('--average-value',
                    type=str,
                    nargs='?',
                    default=None,
                    help='Average image value')
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
parser.add_argument('--preproc',
                    type=str,
                    nargs='?',
                    default='caffe',
                    help='Variant of image preprocessing to use (use ? to get a list)')
parser.add_argument('--remove-dropout',
                    dest='remove_dropout',
                    action='store_true',
                    help='Remove dropout layers')
parser.add_argument('--no-remove-dropout',
                    dest='remove_dropout',
                    action='store_false',
                    help='Do not remove dropout layers')
parser.add_argument('--remove-loss',
                    dest='remove_loss',
                    action='store_true',
                    help='Remove loss layers')
parser.add_argument('--no-remove-loss',
                    dest='remove_loss',
                    action='store_false',
                    help='Do not remove loss layers')
parser.add_argument('--append-softmax',
                    dest='append_softmax',
                    action='append',
                    default=[],
                    help='Add a softmax layer after the specified layer')
parser.set_defaults(remove_dropout=True)
args = parser.parse_args()

print 'Caffe varaint set to', args.caffe_variant
if args.caffe_variant == 'vgg-caffe':
  import proto.vgg_caffe_pb2 as caffe_pb2
elif args.caffe_variant == 'caffe-old':
  import proto.caffe_old_pb2 as caffe_pb2
elif args.caffe_variant == 'caffe':
  import proto.caffe_pb2 as caffe_pb2
elif args.caffe_variant == 'caffe_0115':
  import proto.caffe_0115_pb2 as caffe_pb2
elif args.caffe_variant == '?':
  print 'Supported variants: caffe, cafe-old, vgg-caffe'
  sys.exit(0)
else:
  print 'Unknown Caffe variant', args.caffe_variant
  sys.exit(1)

if args.preproc == '?':
  print 'Preprocessing variants: caffe, vgg'
  sys.exit(0)
if args.preproc not in ['caffe', 'vgg-caffe']:
  print 'Unknown preprocessing variant', args.preproc
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
      floor((size[0] + pad[0]+pad[1] - filter_support[0]) / stride[0]) + 1, \
      floor((size[1] + pad[2]+pad[3] - filter_support[1]) / stride[1]) + 1]

def row(x):
  return np.array(x,dtype=float).reshape(1,-1)

def rowcell(x):
  return np.array(x,dtype=object).reshape(1,-1)

def bilinear_interpolate(im, x, y):
  x = np.asarray(x)
  y = np.asarray(y)

  x0 = np.floor(x).astype(int)
  x1 = x0 + 1
  y0 = np.floor(y).astype(int)
  y1 = y0 + 1

  x0 = np.clip(x0, 0, im.shape[1]-1);
  x1 = np.clip(x1, 0, im.shape[1]-1);
  y0 = np.clip(y0, 0, im.shape[0]-1);
  y1 = np.clip(y1, 0, im.shape[0]-1);

  Ia = im[ y0, x0 ]
  Ib = im[ y1, x0 ]
  Ic = im[ y0, x1 ]
  Id = im[ y1, x1 ]

  wa = (1-x+x0) * (1-y+y0)
  wb = (1-x+x0) * (y-y0)
  wc = (x-x0) * (1-y+y0)
  wd = (x-x0) * (y-y0)

  wa = wa.reshape(x.shape[0], x.shape[1], 1)
  wb = wb.reshape(x.shape[0], x.shape[1], 1)
  wc = wc.reshape(x.shape[0], x.shape[1], 1)
  wd = wd.reshape(x.shape[0], x.shape[1], 1)

  return wa*Ia + wb*Ib + wc*Ic + wd*Id

# --------------------------------------------------------------------
#                                                   Load average image
# --------------------------------------------------------------------

average_image = None
if args.average_image:
  print 'Loading average image from {}'.format(args.average_image.name)
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
elif args.average_value:
  rgb = make_tuple(args.average_value)
  print 'Using average image value', rgb
  # this will be resized later to a constant image
  average_image = np.array(rgb,dtype=float).reshape(1,1,3,order='F')

# --------------------------------------------------------------------
#                                                        Load synseths
# --------------------------------------------------------------------

synsets_wnid=None
synsets_name=None
if args.synsets:
  print 'Loading synsets from {}'.format(args.synsets.name)
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

# Caffe stores the network structure and data into two different files
# We load them both and merge into a single MATLAB structure

print 'Loading Caffe CNN configuration from {}'.format(args.caffe_params.name)
net=caffe_pb2.NetParameter()
google.protobuf.text_format.Merge(args.caffe_params.read(), net)

print 'Loading Caffe CNN parameters from {}'.format(args.caffe_data.name)
data=caffe_pb2.NetParameter()
data.MergeFromString(args.caffe_data.read())

# Fix legacy format
if args.caffe_variant in ['vgg-caffe', 'caffe-old']:
  layers = [x.layer for x in net.layers]
else:
  layers = net.layers

# --------------------------------------------------------------------
#                                                       Convert layers
# --------------------------------------------------------------------

print 'Converting {} layers'.format(len(layers))

if len(net.input_dim) > 0:
  data_size = [net.input_dim[2],
               net.input_dim[3],
               net.input_dim[1]]
else:
  layer = find(layers, 'data')
  data_size = [layer.transform_param.crop_size,
               layer.transform_param.crop_size,
               3]

# variables
mlayerdt = [('name',object),
            ('type',object),
            ('inputs',object),
            ('outputs',object),
            ('params',object),
            ('block',object)]
mparamdt = [('name',object),
            ('value',object)]
mnet = {'layers': np.empty((0,), dtype=mlayerdt),
        'params': np.empty((0,), dtype=mparamdt)}
vars = {'data': data_size, 'label': [1,1,1]}
params = {}

# scan all layers in net_param
from_var_redirect = {}
def from_redirect(names):
  return [from_var_redirect[x] if from_var_redirect.has_key(x) else x for x in names]

def getopts(layer, name):
  if hasattr(layer, name):
    return getattr(layer, name)
  else:
    return layer

for layer in layers:

  # get the type of layer
  # depending on the Caffe variant, this is a string or a numeric
  # ID, which we convert back to a string
  ltype = layer.type
  if not isinstance(ltype, basestring): ltype = layers_type[ltype]
  print 'Processing layer {} of type \'{}\''.format(layer.name, ltype)

  # search for a corresponding layer in net_data
  arrays = []
  support = [1,1]
  pad = [0,0,0,0]
  stride = [1,1]
  inputs = from_redirect(layer.bottom)
  outputs = layer.top
  params = []

  print '  Inputs:'
  for i in inputs:
    var = vars[i]
    print '    {} [{} {} {}]'.format(i, var[0], var[1], var[2])
  print '  Outputs:'
  for o in outputs:
    print '    {}'.format(o)

  dlayer = find(data.layers,layer.name)
  if dlayer:
    for b in dlayer.blobs:
      arrays.append(blobproto_to_array(b).astype('float32'))
      print '  Blob of size', arrays[-1].shape

  mtype = None
  mopts = {}
  mparam = np.empty((0,),dtype=mparamdt)

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if ltype == 'conv':
    if len(inputs) > 1: sys.exit('Invalid number of inputs')

    # extract options (which depends on the Caffe version)
    opts = getopts(layer, 'convolution_param')
    if hasattr(layer, 'kernelsize'):
      support = [opts.kernelsize]*2
    else:
      support = [opts.kernel_size]*2
    pad = [opts.pad]*4
    stride = [opts.stride]*2

    if args.transpose:
      arrays[0] = arrays[0].transpose([1,0,2,3])
      if inputs[0] == 'data':
        # BGR -> RGB
        arrays[0] = arrays[0][:,:,: : -1,:]

    mtype = u'dagnn.Conv'
    mopts['size'] = row(arrays[0].shape)
    mopts['pad'] = row(pad)
    mopts['stride'] = row(stride)

    params = [layer.name + 'f', layer.name + 'b']
    mparam = np.empty((2,),dtype=mparamdt)
    mparam['name'] = params
    mparam['value'] = arrays

    vars[outputs[0]] = \
        get_output_size(vars[inputs[0]],
                        support, pad, stride) + [opts.num_output]

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype == 'relu':
    if len(inputs) > 1: sys.exit('Invalid number of inputs')
    mtype = u'dagnn.ReLU'
    if inputs[0] == outputs[0]:
      print '  Separating in-place ReLU'
      outputs[0] = outputs[0] + 'relu'
      from_var_redirect[inputs[0]] = outputs[0]
    vars[outputs[0]] = vars[inputs[0]]

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype == 'lrn':
    if len(inputs) > 1: sys.exit('Invalid number of inputs')
    opts = getopts(layer, 'lrn_param')
    local_size = float(opts.local_size)
    alpha = float(opts.alpha)
    beta = float(opts.beta)
    kappa = opts.k if hasattr(opts,'k') else 1.

    mtype = u'dagnn.LRN'
    mopts['param'] = row([local_size, kappa, alpha/local_size, beta])
    vars[outputs[0]] = vars[inputs[0]]

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype == 'pool':
    if len(inputs) > 1: sys.exit('Invalid number of inputs')
    opts = getopts(layer, 'pooling_param')
    if hasattr(layer, 'kernelsize'):
      support = [opts.kernelsize]*2
    else:
      support = [opts.kernel_size]*2
    pad = [opts.pad]*4
    stride = [opts.stride]*2

    # adjust the padding for compatibility with MatConvNet
    # unforunately the adjustment depends on the data size
    size = vars[inputs[0]]
    pad[1] += ceil((size[0] - support[0])/float(stride[0]))*stride[0] \
              + support[0] - size[0]
    pad[3] += ceil((size[1] - support[1])/float(stride[1]))*stride[1] \
              + support[1] - size[1]

    mtype = u'dagnn.Pooling'
    mopts['poolSize'] = row(support)
    mopts['method'] = ['max', 'avg'][opts.pool]
    mopts['pad'] = row(pad)
    mopts['stride'] = row(stride)
    vars[outputs[0]] = get_output_size(size, support, pad, stride) + [size[2]]

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype == 'innerproduct' or ltype == 'inner_product':
    if len(inputs) > 1: sys.exit('Invalid number of inputs')
    opts = getopts(layer, 'inner_product_param')
    size = vars[inputs[0]]
    support = [size[0], size[1]]
    pad = [0]*4
    stride = [1]*2

    arrays[0] = arrays[0].reshape(
      size[0],
      size[1],
      size[2],
      opts.num_output,
      order='F')
    arrays[1] = np.squeeze(arrays[1], (2,3))

    if args.transpose:
      arrays[0] = arrays[0].transpose([1,0,2,3])
      if inputs[0] == 'data':
        # BGR -> RGB
        arrays[0] = arrays[0][:,:,: : -1,:]

    mtype = u'dagnn.Conv'
    mopts['size'] = row(arrays[0].shape)
    mopts['pad'] = row(pad)
    mopts['stride'] = row(stride)

    params = [layer.name + 'f', layer.name + 'b']
    mparam = np.empty((2,),dtype=mparamdt)
    mparam['name'] = params
    mparam['value'] = arrays

    vars[outputs[0]] = get_output_size(size, support, pad, stride) + [opts.num_output]

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype == 'dropout':
    if len(inputs) > 1: sys.exit('Invalid number of inputs')
    if args.remove_dropout:
      print '   Removing dropout layer, creating map {} -> {}'.format(outputs[0], inputs[0])
      from_var_redirect[outputs[0]] = inputs[0]
      continue

    opts = getopts(layer, 'dropout_param')
    mtype = u'dagnn.DropOut'
    mopts['rate'] = float(opts.dropout_ratio)
    vars[outputs[0]] = vars[inputs[0]]

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype == 'softmax':
    if len(inputs) > 1: sys.exit('Invalid number of inputs')
    mtype = u'dagnn.SoftMax'
    vars[outputs[0]] = vars[inputs[0]]

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype == 'softmax_loss':
    if args.remove_loss:
      print '   Removing loss layer, creating map {} -> {}'.format(outputs[0], inputs[0])
      from_var_redirect[outputs[0]] = inputs[0]
      continue

    mtype = u'dagnn.Loss'
    mopts['loss'] = 'softmaxlog'
    vars[outputs[0]] = vars[inputs[0]][0:2] + [1]

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype == 'concat':
    opts = getopts(layer, 'concat_param')
    size = vars[inputs[0]]
    if opts.concat_dim > 0 and opts.concat_dim < 4:
      size = vars[inputs[0]]
      cdim = [-1, 2, 0, 1][opts.concat_dim]
      size[cdim] = sum([vars[i][cdim] for i in inputs])
    elif not opts.concat_dim == 0:
      # 0 implies concatenating the fourth dimension (images) and it is ok
      sys.exit('Invalid concat dimension')

    mtype = u'dagnn.Concat'
    mopts['dim'] = [4, 3, 1, 2][opts.concat_dim]
    vars[outputs[0]] = size

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype == 'data':
    # Do not insert data layers
    print('  Skipping')
    continue

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype == 'accuracy':
    # Do not insert accuracy layer
    print('  Skipping')
    continue

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  else:
    mtype = ltype
    vars[outputs[0]] = vars[inputs[0]]
    print 'Warning: unknown layer type', ltype

  print '  Support:', support
  print '  Pad:', pad
  print '  Stride:', stride

  mlayer = np.empty((1,),dtype=mlayerdt)
  mlayer['name'][0] = layer.name
  mlayer['type'][0] = mtype
  mlayer['inputs'][0] = rowcell(inputs)
  mlayer['outputs'][0] = rowcell(outputs)
  mlayer['params'][0] = rowcell(params)
  mlayer['block'][0] = dict_to_struct_array(mopts)

  mnet['layers'] = np.append(mnet['layers'], mlayer)
  mnet['params'] = np.append(mnet['params'], mparam)

# --------------------------------------------------------------------
#                                                       Append softmax
# --------------------------------------------------------------------

for i, name in enumerate(args.append_softmax):
  # searh for a layer of the specified name
  l = next((i for (i, l) in enumerate(layers) if l.name == name), None)
  if l is None:
    print 'Cannot append softmax to layer {} as no such layer could be found'.format(name)
    sys.exit(1)

  if len(args.append_softmax) > 1:
    layerName = 'softmax' + (i + 1)
    outputs= ['prediction' + (i + 1)]
  else:
    layerName = 'softmax'
    outputs = ['prediction']
  inputs = from_redirect(layers[l].top[0:1])

  print 'Appending softmax layer \'{}\' after layer \'{}\''.format(layerName, name)
  print '  Inputs:'
  for i in inputs:
    var = vars[i]
    print '    {} [{} {} {}]'.format(i, var[0], var[1], var[2])
  print '  Outputs:'
  for o in outputs:
    print '    {}'.format(o)

  mlayer = np.empty((1,),dtype=mlayerdt)
  mlayer['name'][0] = layerName
  mlayer['type'][0] = u'dagnn.SoftMax'
  mlayer['inputs'][0] = rowcell(inputs)
  mlayer['outputs'][0] = rowcell(outputs)
  mlayer['params'][0] = rowcell([])
  mlayer['block'][0] = dict_to_struct_array({})
  vars[outputs[0]] = vars[inputs[0]]

  mnet['layers'] = np.append(mnet['layers'], mlayer)

# --------------------------------------------------------------------
#                                                        Normalization
# --------------------------------------------------------------------

size = vars['data']

mkn = {}
mkn['imageSize'] = row(size)

if average_image is not None:
  x = numpy.linspace(0, average_image.shape[1]-1, size[0])
  y = numpy.linspace(0, average_image.shape[0]-1, size[1])
  x, y = np.meshgrid(x, y, sparse=False, indexing='xy')
  mkn['averageImage']=bilinear_interpolate(average_image, x, y)
else:
  mkn['averageImage']=np.zeros((0,),dtype='float')

if args.preproc == 'caffe':
  mkn['interpolation'] = 'bicubic'
  mkn['keepAspect'] = False
  mkn['border'] = row([256 - size[0], 256 - size[1]])
else:
  mkn['interpolation'] = 'bilinear'
  mkn['keepAspect'] = True
  mkn['border'] = row([0, 0])

# --------------------------------------------------------------------
#                                                              Classes
# --------------------------------------------------------------------

meta = {'normalization': mkn}

meta_classes = {}
if synsets_wnid: meta_classes['name'] = np.array(synsets_wnid, dtype=np.object).reshape(1,-1)
if synsets_name: meta_classes['description'] = np.array(synsets_name, dtype=np.object).reshape(1,-1)
if meta_classes:
  meta['classes'] = meta_classes
if meta:
  mnet['meta'] = meta

# --------------------------------------------------------------------
#                                                          Save output
# --------------------------------------------------------------------

print 'Saving network to {}'.format(args.output.name)
scipy.io.savemat(args.output, mnet, oned_as='column')
