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
from layers import *

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

# --------------------------------------------------------------------
#                                                        Parse options
# --------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Convert a Caffe CNN into a MATLAB structure.')
parser.add_argument('caffe_proto',
                    type=argparse.FileType('rb'),
                    help='The Caffe CNN parameter file (ASCII .proto)')
parser.add_argument('--caffe-data',
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
parser.add_argument('--class-names',
                    type=str,
                    nargs='?',
                    help='Class names')
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
parser.add_argument('--color-format',
                    dest='color_format',
                    default='bgr',
                    action='store',
                    help='Set the color format used by the network: ''rgb'' or ''bgr'' (default)')
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
parser.add_argument('--output-format',
                    dest='output_format',
                    default='dagnn',
                    help='Either ''dagnn'' or ''simplenn''')

parser.set_defaults(transpose=True)
parser.set_defaults(remove_dropout=False)
parser.set_defaults(remove_loss=False)
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
elif args.caffe_variant == 'caffe_6e3916':
  import proto.caffe_6e3916_pb2 as caffe_pb2
elif args.caffe_variant == '?':
  print 'Supported variants: caffe, cafe-old, caffe_0115, caffe_6e3916, vgg-caffe'
  sys.exit(0)
else:
  print 'Unknown Caffe variant', args.caffe_variant
  sys.exit(1)

if args.preproc == '?':
  print 'Preprocessing variants: caffe, vgg, fcn'
  sys.exit(0)
if args.preproc not in ['caffe', 'vgg-caffe', 'fcn']:
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

# Get the parameters for a layer from Caffe's proto entries
def getopts(layer, name):
  if hasattr(layer, name):
    return getattr(layer, name)
  else:
    # Older Caffe proto formats did not have sub-structures for layer
    # specific parameters but mixed everything up! This falls back to
    # that situation when fetching the parameters.
    return layer

# --------------------------------------------------------------------
#                                                   Load average image
# --------------------------------------------------------------------

average_image = None
resize_average_image = False
if args.average_image:
  print 'Loading average image from {}'.format(args.average_image.name)
  resize_average_image = True # in case different from data size
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

if args.average_value:
  rgb = make_tuple(args.average_value)
  print 'Using average image value', rgb
  # this will be resized later to a constant image
  average_image = np.array(rgb,dtype=float).reshape(1,1,3,order='F')
  resize_average_image = False

# --------------------------------------------------------------------
#                                      Load ImageNet synseths (if any)
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

if args.class_names:
  synsets_wnid=list(make_tuple(args.class_names))
  synsets_name=synsets_wnid

# --------------------------------------------------------------------
#                                                          Load layers
# --------------------------------------------------------------------

# Caffe stores the network structure and data into two different files
# We load them both and merge them into a single MATLAB structure

net=caffe_pb2.NetParameter()
data=caffe_pb2.NetParameter()

print 'Loading Caffe CNN structure from {}'.format(args.caffe_proto.name)
google.protobuf.text_format.Merge(args.caffe_proto.read(), net)

if args.caffe_data:
  print 'Loading Caffe CNN parameters from {}'.format(args.caffe_data.name)
  data.MergeFromString(args.caffe_data.read())

# --------------------------------------------------------------------
#                                   Read layers in a CaffeModel object
# --------------------------------------------------------------------

print 'Converting {} layers'.format(len(net.layers))

cmodel = CaffeModel()
for layer in net.layers:

  # Depending on how old the proto-buf, the top and bottom parameters
  # are found at a different level than the others
  top = layer.top
  bottom = layer.bottom
  if args.caffe_variant in ['vgg-caffe', 'caffe-old']:
    layer = layer.layer

  # get the type of layer
  # depending on the Caffe variant, this is a string or a numeric
  # ID, which we convert back to a string
  ltype = layer.type
  if not isinstance(ltype, basestring): ltype = layers_type[ltype]
  print 'Processing layer {} of type \'{}\''.format(layer.name, ltype)

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if ltype == 'conv' or ltype == 'deconvolution':
    opts = getopts(layer, 'convolution_param')
    if hasattr(layer, 'kernelsize'):
      kernelSize = [opts.kernelsize]*2
    else:
      kernelSize = [opts.kernel_size]*2
    if hasattr(layer, 'bias_term'):
      bias_term = opts.bias_term
    else:
      bias_term = True
    pad = [opts.pad]*4
    stride = [opts.stride]*2
    if ltype == 'conv':
      clayer = CaffeConv(layer.name,
                         bottom,
                         top,
                         kernelSize,
                         bias_term,
                         opts.num_output,
                         opts.group,
                         [opts.stride] * 2,
                         [opts.pad] * 4)
    else:
      clayer = CaffeDeconvolution(layer.name,
                                  bottom,
                                  top,
                                  kernelSize,
                                  bias_term,
                                  opts.num_output,
                                  opts.group,
                                  [opts.stride] * 2,
                                  [opts.pad] * 4)

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype == 'innerproduct' or ltype == 'inner_product':
    opts = getopts(layer, 'inner_product_param')
    #assert(opts.axis == 1)
    if hasattr(layer, 'bias_term'):
      bias_term = opts.bias_term
    else:
      bias_term = True
    clayer = CaffeInnerProduct(layer.name,
                               bottom,
                               top,
                               bias_term,
                               opts.num_output)

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype == 'relu':
    clayer = CaffeReLU(layer.name,
                       bottom,
                       top)


  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype == 'crop':
    clayer = CaffeCrop(layer.name,
                       bottom,
                       top)

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype == 'lrn':
    opts = getopts(layer, 'lrn_param')
    local_size = float(opts.local_size)
    alpha = float(opts.alpha)
    beta = float(opts.beta)
    kappa = opts.k if hasattr(opts,'k') else 1.
    clayer = CaffeLRN(layer.name,
                      bottom,
                      top,
                      local_size,
                      kappa,
                      alpha,
                      beta)

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype == 'pool':
    opts = getopts(layer, 'pooling_param')
    if hasattr(layer, 'kernelsize'):
      kernelSize = [opts.kernelsize]*2
    else:
      kernelSize = [opts.kernel_size]*2
    clayer = CaffePooling(layer.name,
                          bottom,
                          top,
                          ['max', 'avg'][opts.pool],
                          kernelSize,
                          [opts.stride]*2,
                          [opts.pad]*4)

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype == 'dropout':
    opts = getopts(layer, 'dropout_param')
    clayer = CaffeDropout(layer.name,
                          bottom,
                          top,
                          opts.dropout_ratio)

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype == 'softmax':
    clayer = CaffeSoftMax(layer.name,
                          bottom,
                          top)

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype == 'softmax_loss':
    clayer = CaffeSoftMaxLoss(layer.name,
                              bottom,
                              top)

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype == 'concat':
    opts = getopts(layer, 'concat_param')
    clayer = CaffeConcat(layer.name,
                         bottom,
                         top,
                         3 - opts.concat_dim) # todo: depreceted in recent Caffes

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype == 'eltwise':
    opts = getopts(layer, 'eltwise_param')
    clayer = CaffeEltWise(layer.name,
                          bottom,
                          top,
                          ['prod', 'sum', 'max'][opts.operation],
                          opts.coeff,
                          opts.stable_prod_grad)

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype == 'data':
    continue

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype == 'accuracy':
    continue

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  else:
    print 'Warning: unknown layer type', ltype
    continue

  clayer.model = cmodel
  cmodel.addLayer(clayer)

  # Fill parameters
  for dlayer in data.layers:
    if args.caffe_variant in ['vgg-caffe', 'caffe-old']:
      dlayer = dlayer.layer
    if dlayer.name == layer.name:
      for i, blob in enumerate(dlayer.blobs):
        array = blobproto_to_array(blob).astype('float32')
        cmodel.params[clayer.params[i]].value = array
        print '  Found parameter blob of size', array.shape

# --------------------------------------------------------------------
#                             Get the size of the input to the network
# --------------------------------------------------------------------

if len(net.input_dim) > 0:
  dataSize = [net.input_dim[2],
              net.input_dim[3],
              net.input_dim[1],
              1]
else:
  layer = find(net.layers, 'data')
  if layer is None:
    print "Warning: could not determine the input data size"
  else:
    dataSize = [layer.transform_param.crop_size,
                layer.transform_param.crop_size,
                3,
                1]

dataVarName = 'data'
if not cmodel.vars.has_key('data'):
  dataVarName = cmodel.layers.elements().next().inputs[0]
cmodel.vars[dataVarName].size = dataSize

# mark data as BGR for the purpose of transposition
# rare Caffe networks are trained in RGB format, so this can be skipped
# this is decided based on the value of the --color-format option
cmodel.vars[dataVarName].bgrInput = (args.color_format == 'bgr')

# --------------------------------------------------------------------
#                                                      Edit operations
# --------------------------------------------------------------------

# May perform several adjustments that depend on the input size:
#
# * For pooling, fix incompatibility between pooling padding in MatConvNet and Caffe
# * For FCNs, compute the amount of crop

cmodel.reshape()

# Transpose to accomodate MATLAB H x W image order

if args.transpose:
  cmodel.transpose()

def escape(name):
  return name.replace('-','_')

# Rename layers, parametrs, and variables if they contain
# symbols that are incompatible with MatConvNet

layerNames = cmodel.layers.keys()
for name in layerNames:
  ename = escape(name)
  if ename == name: continue
  # ensure unique
  while cmodel.layers.has_key(ename): ename = ename + 'x'
  print "Renaming layer {} to {}".format(name, ename)
  cmodel.renameLayer(name, ename)

varNames = cmodel.vars.keys()
for name in varNames:
  ename = escape(name)
  if ename == name: continue
  while cmodel.vars.has_key(ename): ename = ename + 'x'
  print "Renaming variable {} to {}".format(name, ename)
  cmodel.renameVar(name, ename)

parNames = cmodel.params.keys()
for name in parNames:
  ename = escape(name)
  if ename == name: continue
  while cmodel.params.has_key(ename): ename = ename + 'x'
  print "Renaming parameter {} to {}".format(name, ename)
  cmodel.renameParam(name, ename)

# Split in-place layers
for layer in cmodel.layers.itervalues():
  if len(layer.inputs[0]) >= 1 and \
        len(layer.outputs[0]) >= 1 and \
        layer.inputs[0] == layer.outputs[0]:
    name = layer.inputs[0]
    ename = layer.inputs[0]
    while cmodel.vars.has_key(ename): ename = ename + 'x'
    print "Splitting in-place: renaming variable {} to {}".format(name, ename)
    cmodel.addVar(ename)
    cmodel.renameVar(name, ename, afterLayer=layer.name)
    layer.inputs[0] = name
    layer.outputs[0] = ename

# Remove dropout
if args.remove_dropout:
  layerNames = cmodel.layers.keys()
  for name in layerNames:
    layer = cmodel.layers[name]
    if type(layer) is CaffeDropout:
      print "Removing dropout layer ", name
      cmodel.renameVar(layer.outputs[0], layer.inputs[0])
      cmodel.removeLayer(name)

# Remove loss
if args.remove_dropout:
  layerNames = cmodel.layers.keys()
  for name in layerNames:
    layer = cmodel.layers[name]
    if type(layer) is CaffeSoftMaxLoss:
      print "Removing loss layer ", name
      cmodel.renameVar(layer.outputs[0], layer.inputs[0])
      cmodel.removeLayer(name)

# Append softmax
for i, name in enumerate(args.append_softmax):
  # search for the layer to append SoftMax to
  if not cmodel.layers.has_key(name):
    print 'Cannot append softmax to layer {} as no such layer could be found'.format(name)
    sys.exit(1)

  if len(args.append_softmax) > 1:
    layerName = 'softmax' + (l + 1)
    outputs= ['prob' + (l + 1)]
  else:
    layerName = 'softmax'
    outputs = ['prob']

  cmodel.addLayer(CaffeSoftMax(layerName,
                               cmodel.layers[name].outputs[0:1],
                               outputs))

cmodel.display()

# --------------------------------------------------------------------
#                                                        Normalization
# --------------------------------------------------------------------

if average_image is not None:
  if resize_average_image:
    x = numpy.linspace(0, average_image.shape[1]-1, dataSize[0])
    y = numpy.linspace(0, average_image.shape[0]-1, dataSize[1])
    x, y = np.meshgrid(x, y, sparse=False, indexing='xy')
    average_image = bilinear_interpolate(average_image, x, y)
else:
  average_image = np.zeros((0,),dtype='float')

mnormalization = {
  'imageSize': row(dataSize),
  'averageImage': average_image,
  'interpolation': 'bilinear',
  'keepAspect': True,
  'border': row([0,0])}

if args.preproc == 'caffe':
  mnormalization['interpolation'] = 'bicubic'
  mnormalization['keepAspect'] = False
  mnormalization['border'] = row([256 - dataSize[0], 256 - dataSize[1]])

# --------------------------------------------------------------------
#                                                              Classes
# --------------------------------------------------------------------

mclassnames = np.empty((0,), dtype=np.object)
mclassdescriptions = np.array((0,), dtype=np.object)

if synsets_wnid:
  mclassnames = np.array(synsets_wnid, dtype=np.object).reshape(1,-1)

if synsets_name:
  mclassdescriptions = np.array(synsets_name, dtype=np.object).reshape(1,-1)

mclasses = dictToMatlabStruct({'name': mclassnames,
                               'description': mclassdescriptions})

# --------------------------------------------------------------------
#                                                    Convert to MATLAB
# --------------------------------------------------------------------

# net.meta
mmeta = dictToMatlabStruct({'normalization': mnormalization,
                            'classes': mclasses})

if args.output_format == 'dagnn':

  # This object should stay a dictionary and not a NumPy array due to
  # how NumPy saves to MATLAB

  mnet = {'layers': np.empty(shape=[0,], dtype=mlayerdt),
          'params': np.empty(shape=[0,], dtype=mparamdt),
          'meta': mmeta}

  for layer in cmodel.layers.itervalues():
    mnet['layers'] = np.append(mnet['layers'], layer.toMatlab(), axis=0)

  for param in cmodel.params.itervalues():
    mnet['params'] = np.append(mnet['params'], param.toMatlab(), axis=0)

  # to row
  mnet['layers'] = mnet['layers'].reshape(1,-1)
  mnet['params'] = mnet['params'].reshape(1,-1)

elif args.output_format == 'simplenn':

  # This object should stay a dictionary and not a NumPy array due to
  # how NumPy saves to MATLAB

  mnet = {'layers': np.empty(shape=[0,], dtype=np.object),
          'meta': mmeta}

  for layer in cmodel.layers.itervalues():
    mnet['layers'] = np.append(mnet['layers'], np.object)
    mnet['layers'][-1] = dictToMatlabStruct(layer.toMatlabSimpleNN())

  # to row
  mnet['layers'] = mnet['layers'].reshape(1,-1)

# --------------------------------------------------------------------
#                                                          Save output
# --------------------------------------------------------------------

print 'Saving network to {}'.format(args.output.name)
scipy.io.savemat(args.output, mnet, oned_as='column')
