# file: layers.py
# brief: A number of objects to wrap caffe layers for conversion
# author: Andrea Vedaldi

from collections import OrderedDict
from math import floor, ceil
import numpy as np
from numpy import array
import scipy
import scipy.io
import scipy.misc
import copy
import collections

layers_type = {}
layers_type[0]  = 'none'
layers_type[1]  = 'accuracy'
layers_type[2]  = 'bnll'
layers_type[3]  = 'concat'
layers_type[4]  = 'conv'
layers_type[5]  = 'data'
layers_type[6]  = 'dropout'
layers_type[7]  = 'euclidean_loss'
layers_type[8]  = 'flatten'
layers_type[9]  = 'hdf5_data'
layers_type[10] = 'hdf5_output'
layers_type[28] = 'hinge_loss'
layers_type[11] = 'im2col'
layers_type[12] = 'image_data'
layers_type[13] = 'infogain_loss'
layers_type[14] = 'inner_product'
layers_type[15] = 'lrn'
layers_type[25] = 'eltwise'
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
layers_type[39] = 'deconvolution'
layers_type[40] = 'crop'

def getFilterOutputSize(size, kernelSize, stride, pad):
    return [floor((size[0] + pad[0]+pad[1] - kernelSize[0]) / stride[0]) + 1., \
            floor((size[1] + pad[2]+pad[3] - kernelSize[1]) / stride[1]) + 1.]

def getFilterTransform(ks, stride, pad):
    y1 = 1. - pad[0] ;
    y2 = 1. - pad[0] + ks[0] - 1 ;
    x1 = 1. - pad[2] ;
    x2 = 1. - pad[2] + ks[1] - 1 ;
    h = y2 - y1 + 1. ;
    w = x2 - x1 + 1. ;
    return CaffeTransform([h, w], stride, [(y1+y2)/2, (x1+x2)/2])

def reorder(aList, order):
    return [aList[i] for i in order]

def row(x):
  return np.array(x,dtype=float).reshape(1,-1)

def rowarray(x):
  return x.reshape(1,-1)

def rowcell(x):
    return np.array(x,dtype=object).reshape(1,-1)

def dictToMatlabStruct(d):
  if not d:
    return np.zeros((0,))
  dt = []
  for x in d.keys():
      pair = (x,object)
      if isinstance(d[x], np.ndarray): pair = (x,type(d[x]))
      dt.append(pair)
  y = np.empty((1,),dtype=dt)
  for x in d.keys():
    y[x][0] = d[x]
  return y

# --------------------------------------------------------------------
#                                                  MatConvNet in NumPy
# --------------------------------------------------------------------

mlayerdt = [('name',object),
            ('type',object),
            ('inputs',object),
            ('outputs',object),
            ('params',object),
            ('block',object)]

mparamdt = [('name',object),
            ('value',object)]

# --------------------------------------------------------------------
#                                                      Vars and params
# --------------------------------------------------------------------

class CaffeBuffer(object):
    def __init__(self, name):
        self.name = name
        self.size = None
        self.value = np.zeros(shape=(0,0), dtype='float32')
        self.bgrInput = False

    def toMatlab(self):
        mparam = np.empty(shape=[1,], dtype=mparamdt)
        mparam['name'][0] = self.name
        mparam['value'][0] = self.value
        return mparam

    def toMatlabSimpleNN(self):
        return self.value

class CaffeTransform(object):
    def __init__(self, size, stride, offset):
        self.size = size
        self.stride = stride
        self.offset = offset

    def __str__(self):
        return "<%s %s %s>" % (self.size, self.stride, self.offset)

def composeTransforms(a, b):
    size = [0.,0.]
    stride = [0.,0.]
    offset = [0.,0.]
    for i in [0,1]:
        size[i] = a.stride[i] * (b.size[i] - 1) + a.size[i]
        stride[i] = a.stride[i] * b.stride[i]
        offset[i] = a.stride[i] * (b.offset[i] - 1) + a.offset[i]
    c = CaffeTransform(size, stride, offset)
    return c

def transposeTransform(a):
    size = [0.,0.]
    stride = [0.,0.]
    offset = [0.,0.]
    for i in [0,1]:
        size[i] = (a.size[i] + a.stride[i] - 1.0) / a.stride[i]
        stride[i] = 1.0/a.stride[i]
        offset[i] = (1.0 + a.stride[i] - a.offset[i]) / a.stride[i]
    c = CaffeTransform(size, stride, offset)
    return c

# --------------------------------------------------------------------
#                                                               Errors
# --------------------------------------------------------------------

class ConversionError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

# --------------------------------------------------------------------
#                                                         Basic Layers
# --------------------------------------------------------------------

class CaffeLayer(object):
    def __init__(self, name, inputs, outputs):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.params = []
        self.model = None

    def reshape(self, model):
        pass

    def display(self):
        print "Layer ", self.name
        print "  Type: %s" % (self.__class__.__name__)
        print "  Inputs: %s" % (self.inputs,)
        print "  Outputs: %s" % (self.outputs,)
        print "  Params: %s" % (self.params,)

    def getTransforms(self, model):
        transforms = []
        for i in enumerate(self.inputs):
            row = []
            for j in enumerate(self.outputs):
                row.append(CaffeTransform([1.,1.], [1.,1.], [1.,1.]))
            transforms.append(row)
        return transforms

    def transpose(self, model):
        pass

    def toMatlab(self):
        mlayer = np.empty(shape=[1,],dtype=mlayerdt)
        mlayer['name'][0] = self.name
        mlayer['type'][0] = None
        mlayer['inputs'][0] = rowcell(self.inputs)
        mlayer['outputs'][0] = rowcell(self.outputs)
        mlayer['params'][0] = rowcell(self.params)
        mlayer['block'][0] = dictToMatlabStruct({})
        return mlayer

    def toMatlabSimpleNN(self):
        mparam = collections.OrderedDict() ;
        mparam['name'] = self.name
        mparam['type'] = None
        return mparam

class CaffeElementWise(CaffeLayer):
    def reshape(self, model):
        for i in range(len(self.inputs)):
            model.vars[self.outputs[i]].size = \
                model.vars[self.inputs[i]].size

class CaffeReLU(CaffeElementWise):
    def __init__(self, name, inputs, outputs):
        super(CaffeReLU, self).__init__(name, inputs, outputs)

    def toMatlab(self):
        mlayer = super(CaffeReLU, self).toMatlab()
        mlayer['type'][0] = u'dagnn.ReLU'
        mlayer['block'][0] = dictToMatlabStruct(
            {'leak': float(0.0) })
        # todo: leak factor
        return mlayer

    def toMatlabSimpleNN(self):
        mlayer = super(CaffeReLU, self).toMatlabSimpleNN()
        mlayer['type'] = u'relu'
        mlayer['leak'] = float(0.0)
        return mlayer

class CaffeLRN(CaffeElementWise):
    def __init__(self, name, inputs, outputs, local_size, kappa, alpha, beta):
        super(CaffeLRN, self).__init__(name, inputs, outputs)
        self.local_size = local_size
        self.kappa = kappa
        self.alpha = alpha
        self.beta = beta

    def toMatlab(self):
        mlayer = super(CaffeLRN, self).toMatlab()
        mlayer['type'][0] = u'dagnn.LRN'
        mlayer['block'][0] = dictToMatlabStruct(
            {'param': row([self.local_size,
                           self.kappa,
                           self.alpha / self.local_size,
                           self.beta])})
        return mlayer

    def toMatlabSimpleNN(self):
        mlayer = super(CaffeLRN, self).toMatlabSimpleNN()
        mlayer['type'] = u'lrn'
        mlayer['param'] = row([self.local_size,
                               self.kappa,
                               self.alpha / self.local_size,
                               self.beta])
        return mlayer

class CaffeSoftMax(CaffeElementWise):
    def __init__(self, name, inputs, outputs):
        super(CaffeSoftMax, self).__init__(name, inputs, outputs)

    def toMatlab(self):
        mlayer = super(CaffeSoftMax, self).toMatlab()
        mlayer['type'][0] = u'dagnn.SoftMax'
        return mlayer

    def toMatlabSimpleNN(self):
        mlayer = super(CaffeSoftMax, self).toMatlabSimpleNN()
        mlayer['type'] = u'softmax'
        return mlayer

class CaffeSoftMaxLoss(CaffeElementWise):
    def __init__(self, name, inputs, outputs):
        super(CaffeSoftMaxLoss, self).__init__(name, inputs, outputs)

    def toMatlab(self):
        mlayer = super(CaffeSoftMaxLoss, self).toMatlab()
        mlayer['type'][0] = u'dagnn.SoftMaxLoss'
        return mlayer

    def toMatlabSimpleNN(self):
        mlayer = super(CaffeSoftMaxLoss, self).toMatlabSimpleNN()
        mlayer['type'] = u'softmax'
        return mlayer

class CaffeDropout(CaffeElementWise):
    def __init__(self, name, inputs, outputs, ratio):
        super(CaffeDropout, self).__init__(name, inputs, outputs)
        self.ratio = ratio

    def toMatlab(self):
        mlayer = super(CaffeDropout, self).toMatlab()
        mlayer['type'][0] = u'dagnn.DropOut'
        mlayer['block'][0] = dictToMatlabStruct({'rate': float(self.ratio)})
        return mlayer

    def toMatlabSimpleNN(self):
        mlayer = super(CaffeDropout, self).toMatlabSimpleNN()
        mlayer['type'] = u'dropout'
        mlayer['rate'] = float(self.ratio)
        return mlayer

    def display(self):
        super(CaffeDropout, self).display()
        print "  Ratio (rate): ", self.ratio

# --------------------------------------------------------------------
#                                                          Convolution
# --------------------------------------------------------------------

class CaffeConv(CaffeLayer):
    def __init__(self, name, inputs, outputs, kernelSize, hasBias, numFilters, numFilterGroups, stride, pad):
        super(CaffeConv, self).__init__(name, inputs, outputs)
        self.params = [name + 'f']
        if hasBias: self.params.append(name + 'b')
        self.hasBias = hasBias
        self.kernelSize = kernelSize
        self.numFilters = numFilters
        self.numFilterGroups = numFilterGroups
        self.filterDimension = None
        self.stride = stride
        self.pad = pad
        self.display()

    def display(self):
        super(CaffeConv, self).display()
        print "  Kernel Size: %s" % self.kernelSize
        print "  Has Bias: %s" % self.hasBias
        print "  Pad: %s" % (self.pad,)
        print "  Stride: %s" % (self.stride,)
        print "  Num Filters: %s" % self.numFilters
        print "  Filter Dimension:", self.filterDimension

    def reshape(self, model):
        varin = model.vars[self.inputs[0]]
        varout = model.vars[self.outputs[0]]
        if len(varin.size) == 0: return
        varout.size = getFilterOutputSize(varin.size[0:2], \
                                              self.kernelSize, self.stride, self.pad) + \
                                              [self.numFilters, varin.size[3]]
        self.filterDimension = varin.size[2] / self.numFilterGroups

    def getTransforms(self, model):
        return [[getFilterTransform(self.kernelSize, self.stride, self.pad)]]

    def transpose(self, model):
        self.kernelSize = reorder(self.kernelSize, [1,0])
        self.stride = reorder(self.stride, [1,0])
        self.pad = reorder(self.pad, [2,3,0,1])
        if model.params[self.params[0]].value.size > 0:
            print "Layer %s transposing filters" % self.name
            param = model.params[self.params[0]]
            param.value = param.value.transpose([1,0,2,3])
            if model.vars[self.inputs[0]].bgrInput:
                print "Layer %s BGR to RGB conversion" % self.name
                param.value = param.value[:,:,: : -1,:]

    def toMatlab(self):
        size = self.kernelSize + [self.filterDimension, self.numFilters]
        mlayer = super(CaffeConv, self).toMatlab()
        mlayer['type'][0] = u'dagnn.Conv'
        mlayer['block'][0] = dictToMatlabStruct(
            {'hasBias': self.hasBias,
             'size': row(size),
             'pad': row(self.pad),
             'stride': row(self.stride)})
        return mlayer

    def toMatlabSimpleNN(self):
        size = self.kernelSize + [self.filterDimension, self.numFilters]
        mlayer = super(CaffeConv, self).toMatlabSimpleNN()
        mlayer['type'] = u'conv'
        mlayer['weights'] = np.empty([1,len(self.params)], dtype=np.object)
        mlayer['size'] = row(size)
        mlayer['pad'] = row(self.pad)
        mlayer['stride'] = row(self.stride)
        for p, name in enumerate(self.params):
            mlayer['weights'][0,p] = self.model.params[name].toMatlabSimpleNN()
        return mlayer

# --------------------------------------------------------------------
#                                                        Inner Product
# --------------------------------------------------------------------

# special case: inner product
class CaffeInnerProduct(CaffeConv):
    def __init__(self, name, inputs, outputs, bias_term, num_outputs):
        ks = [None, None, None, num_outputs]
        super(CaffeInnerProduct, self).__init__(name, inputs, outputs,
                                                ks,
                                                bias_term,
                                                num_outputs, # n filters
                                                1, # n groups
                                                [1, 1], # stride
                                                [0, 0, 0, 0]) # pad

    def reshape(self, model):
        if len(model.vars[self.inputs[0]].size) == 0: return
        s = model.vars[self.inputs[0]].size
        self.kernelSize = [s[0], s[1], s[2], self.numFilters]
        print "Layer %s: inner product converted to filter bank of shape %s" % (self.name, self.kernelSize)
        param = model.params[self.params[0]]
        if param.value.size > 0:
            print "Layer %s: reshaping inner product paramters of shape %s into a filter bank" % (self.name, param.value.shape)
            param.value = param.value.reshape(self.kernelSize, order='F')
        super(CaffeInnerProduct, self).reshape(model)

# --------------------------------------------------------------------
#                                                        Deconvolution
# --------------------------------------------------------------------

class CaffeDeconvolution(CaffeConv):
    def __init__(self, name, inputs, outputs, kernelSize, hasBias, numFilters, numFilterGroups, stride, pad):
        super(CaffeDeconvolution, self).__init__(name, inputs, outputs, kernelSize, hasBias, numFilters, numFilterGroups, stride, pad)

    def display(self):
        super(CaffeDeconvolution, self).display()

    def reshape(self, model):
        if len(model.vars[self.inputs[0]].size) == 0: return
        model.vars[self.outputs[0]].size = \
            getFilterOutputSize(model.vars[self.inputs[0]].size[0:2],
                                self.kernelSize, self.stride, self.pad) + \
            [self.numFilters, model.vars[self.inputs[0]].size[3]]
        self.filterDimension = model.vars[self.inputs[0]].size[2]

    def getTransforms(self, model):
        t = getFilterTransform(self.kernelSize, self.stride, self.pad)
        t = transposeTransform(t)
        return [[t]]

    def transpose(self, model):
        self.kernelSize = reorder(self.kernelSize, [1,0])
        self.stride = reorder(self.stride, [1,0])
        self.pad = reorder(self.pad, [2,3,0,1])
        if model.params[self.params[0]].value.size > 0:
            print "Layer %s transposing filters" % self.name
            param = model.params[self.params[0]]
            param.value = param.value.transpose([1,0,2,3])
            if model.vars[self.inputs[0]].bgrInput:
                print "Layer %s BGR to RGB conversion" % self.name
                param.value = param.value[:,:,:,: : -1]

    def toMatlab(self):
        size = self.kernelSize +  [self.numFilters, \
                                      self.filterDimension / self.numFilterGroups]
        mlayer = super(CaffeDeconvolution, self).toMatlab()
        mlayer['type'][0] = u'dagnn.ConvTranspose'
        mlayer['block'][0] = dictToMatlabStruct(
            {'hasBias': self.hasBias,
             'size': row(size),
             'upsample': row(self.stride),
             'crop': row(self.pad)})
        return mlayer

    def toMatlabSimpleNN(self):
        size = self.kernelSize +  [self.numFilters, \
                                      self.filterDimension / self.numFilterGroups]
        mlayer = super(CaffeDeconvolution, self).toMatlabSimpleNN()
        mlayer['type'] = u'convt'
        mlayer['weights'] = np.empty([1,len(self.params)], dtype=np.object)
        mlayer['size'] = row(size)
        mlayer['upsample'] =  row(self.stride)
        mlayer['crop'] = row(self.pad)
        for p, name in enumerate(self.params):
            mlayer['weights'][0,p] = self.model.params[name].toMatlabSimpleNN()
        return mlayer

# --------------------------------------------------------------------
#                                                              Pooling
# --------------------------------------------------------------------

class CaffePooling(CaffeLayer):
    def __init__(self, name, inputs, outputs, method, kernelSize, stride, pad):
        super(CaffePooling, self).__init__(name, inputs, outputs)
        self.method = method
        self.kernelSize = kernelSize
        self.stride = stride
        self.pad = pad
        self.padCorrected = []

    def display(self):
        super(CaffePooling, self).display()
        print "  Method: ", self.method
        print "  Kernel Size: %s" % self.kernelSize
        print "  Pad: %s" % (self.pad,)
        print "  PadCorrected: %s" % (self.padCorrected,)
        print "  Stride: %s" % (self.stride,)

    def reshape(self, model):
        if len(model.vars[self.inputs[0]].size) == 0: return
        size = model.vars[self.inputs[0]].size
        ks = self.kernelSize
        stride = self.stride
        # MatConvNet uses a slighly different definition of padding, which we think
        # is the correct one (it corresponds to the filters)
        self.padCorrected = copy.deepcopy(self.pad)
        for i in [0, 1]:
            self.padCorrected[1 + i*2] = min(
                self.pad[1 + i*2] + self.stride[i] - 1,
                self.kernelSize[i] - 1)
        model.vars[self.outputs[0]].size = \
            getFilterOutputSize(size[0:2], ks, self.stride, self.padCorrected) + \
            size[2:5]

    def getTransforms(self, model):
        return [[getFilterTransform(self.kernelSize, self.stride, self.pad)]]

    def transpose(self, model):
        self.kernelSize = reorder(self.kernelSize, [1,0])
        self.stride = reorder(self.stride, [1,0])
        self.pad = reorder(self.pad, [2,3,0,1])
        self.padCorrected = reorder(self.padCorrected, [2,3,0,1])

    def toMatlab(self):
        mlayer = super(CaffePooling, self).toMatlab()
        mlayer['type'][0] = u'dagnn.Pooling'
        mlayer['block'][0] = dictToMatlabStruct(
            {'method': self.method,
             'poolSize': row(self.kernelSize),
             'stride': row(self.stride),
             'pad': row(self.padCorrected)})
        return mlayer

    def toMatlabSimpleNN(self):
        mlayer = super(CaffePooling, self).toMatlabSimpleNN()
        mlayer['type'] = u'pool'
        mlayer['method'] = self.method
        mlayer['pool'] = row(self.kernelSize)
        mlayer['stride'] = row(self.stride)
        mlayer['pad'] = row(self.padCorrected)
        return mlayer

# --------------------------------------------------------------------
#                                                        Concatenation
# --------------------------------------------------------------------

class CaffeConcat(CaffeLayer):
    def __init__(self, name, inputs, outputs, concatDim):
        super(CaffeConcat, self).__init__(name, inputs, outputs)
        self.concatDim = concatDim

    def transpose(self, model):
        self.concatDim = [1, 0, 2, 3][self.concatDim]

    def reshape(self, model):
        sizes = [model.vars[x].size for x in self.inputs]
        osize = copy.deepcopy(sizes[0])
        osize[self.concatDim] = 0
        for thisSize in sizes:
            for i in range(len(thisSize)):
                if self.concatDim == i:
                    osize[i] = osize[i] + thisSize[i]
                else:
                    if osize[i] != thisSize[i]:
                        print "Warning: concat layer: inconsistent input dimensions", sizes
        model.vars[self.outputs[0]].size = osize

    def display(self):
        super(CaffeConcat, self).display()
        print "  Concat Dim: ", self.concatDim

    def toMatlab(self):
        mlayer = super(CaffeConcat, self).toMatlab()
        mlayer['type'][0] = u'dagnn.Concat'
        mlayer['block'][0] = dictToMatlabStruct({'dim': float(self.concatDim) + 1})
        return mlayer

    def toMatlabSimpleNN(self):
        raise ConversionError('Concat layers do not work in a SimpleNN network')

# --------------------------------------------------------------------
#                                                   EltWise (Sum, ...)
# --------------------------------------------------------------------

class CaffeEltWise(CaffeElementWise):
    def __init__(self, name, inputs, outputs, operation, coeff, stableProdGrad):
        super(CaffeEltWise, self).__init__(name, inputs, outputs)
        self.operation = operation
        self.coeff = coeff
        self.stableProdGrad = stableProdGrad

    def toMatlab(self):
        mlayer = super(CaffeEltWise, self).toMatlab()
        if self.operation == 'sum':
            mlayer['type'][0] = u'dagnn.Sum'
        else:
            # not implemented
            assert(False)
        return mlayer

    def display(self):
        super(CaffeEltWise, self).display()
        print "  Operation: ", self.operation
        print "  Coeff: %s" % self.coeff
        print "  Stable Prod Grad: %s" % self.stableProdGrad

    def reshape(self, model):
        model.vars[self.outputs[0]].size = \
            model.vars[self.inputs[0]].size
        for i in range(1, len(self.inputs)):
            assert(model.vars[self.inputs[0]].size == model.vars[self.inputs[i]].size)

    def toMatlabSimpleNN(self):
        raise ConversionError('EltWise (sum, ...) layers do not work in a SimpleNN network')

# --------------------------------------------------------------------
#                                                                 Crop
# --------------------------------------------------------------------

class CaffeCrop(CaffeLayer):
    def __init__(self, name, inputs, outputs):
        super(CaffeCrop, self).__init__(name, inputs, outputs)
        self.crop = []

    def display(self):
        super(CaffeCrop, self).display()
        print "  Crop: %s" % self.crop

    def reshape(self, model):
        # this is quite complex as we need to compute on the fly
        # the geometry
        tfs1 = model.getParentTransforms(self.inputs[0], self.name)
        tfs2 = model.getParentTransforms(self.inputs[1], self.name)

        print
        print self.name, self.inputs[0]
        for a,x in enumerate(tfs1): print "%10s %s" % (x,tfs1[x])
        print self.name, self.inputs[1]
        for a,x in enumerate(tfs2): print "%10s %s" % (x,tfs2[x])

        # the goal is to crop inputs[0] to make it as big as inputs[1] and
        # aligned to it; so now we find the map from inputs[0] to inputs[1]

        tf = None
        for name, tf2 in tfs2.items():
            if tfs1.has_key(name):
                tf1 = tfs1[name]
                tf = composeTransforms(transposeTransform(tf2), tf1)
                break
        if tf is None:
            print "Error: could not find common ancestor for inputs '%s' and '%s' of the CaffeCrop layer '%s'" % (self.inputs[0], self.inputs[1], self.name)
            sys.exit(1)
        print "  Transformation %s -> %s = %s" % (self.inputs[0],
                                                  self.inputs[1], tf)
        # for this to make sense it shoudl be tf.stride = 1
        assert(tf.stride[0] == 1 and tf.stride[1] == 1)

        # finally we can get the crops!
        self.crop = [0.,0.]
        for i in [0,1]:
            # i' = alpha (i - 1) + beta + crop = 1 for i = 1
            # crop = 1 - beta
            self.crop[i] =  round(1 - tf.offset[i])
        print "  Crop %s" % self.crop

        # print
        # print "resolved"
        # tfs3 = model.getParentTransforms(self.outputs[0])
        # for a,x in enumerate(tfs3): print "%10s %s" % (x,tfs3[x])

        # now compute output variable size, which will be the size of the second input
        model.vars[self.outputs[0]].size = model.vars[self.inputs[1]].size

    def getTransforms(self, model):
        t = CaffeTransform([1.,1.], [1.,1.], [1.+self.crop[0],1.+self.crop[1]])
        return [[t],[None]]

    def toMatlab(self):
        mlayer = super(CaffeCrop, self).toMatlab()
        mlayer['type'][0] = u'dagnn.Crop'
        mlayer['block'][0] = dictToMatlabStruct({'crop': row(self.crop)})
        return mlayer

    def toMatlabSimpleNN(self):
        # todo: simple 1 input crop layers should be supported though!
        raise ConversionError('Crop layers do not work in a SimpleNN network')

class CaffeData(CaffeLayer):
    def __init__(self, name, inputs, outputs, size):
        super(CaffeData, self).__init__(name, inputs, outputs)
        self.size = size

# --------------------------------------------------------------------
#                                                          Caffe Model
# --------------------------------------------------------------------

class CaffeModel(object):
    def __init__(self):
        self.layers = OrderedDict()
        self.vars = OrderedDict()
        self.params = OrderedDict()

    def addLayer(self, layer):
        ename = layer.name
        while self.layers.has_key(ename):
            ename = ename + 'x'
        if layer.name != ename:
            print "Warning: a layer with name %s was already found, using %s instead" % \
                (layer.name, ename)
            layer.name = ename
        for v in layer.inputs:  self.addVar(v)
        for v in layer.outputs: self.addVar(v)
        for p in layer.params: self.addParam(p)
        self.layers[layer.name] = layer

    def addVar(self, name):
        if not self.vars.has_key(name):
            self.vars[name] = CaffeBuffer(name)

    def addParam(self, name):
        if not self.params.has_key(name):
            self.params[name] = CaffeBuffer(name)

    def renameLayer(self, old, new):
        self.layers[old].name = new
        # reinsert layer with new name -- this mess is to preserve the order
        layers = OrderedDict([(new,v) if k==old else (k,v)
                              for k,v in self.layers.items()])
        self.layers = layers

    def renameVar(self, old, new, afterLayer=None):
        self.vars[old].name = new
        if afterLayer is not None:
            start = self.layers.keys().index(afterLayer) + 1
        else:
            start = 0
        # fix all references to the variable
        for layer in self.layers.values()[start:-1]:
            layer.inputs = [new if x==old else x for x in layer.inputs]
            layer.outputs = [new if x==old else x for x in layer.outputs]
        var = self.vars[old]
        del self.vars[old]
        self.vars[new] = var

    def renameParam(self, old, new):
        self.params[old].name = new
        # fix all references to the variable
        for layer in self.layers.itervalues():
            layer.params = [new if x==old else x for x in layer.params]
        var = self.params[old]
        del self.params[old]
        self.params[new] = var

    def removeParam(self, name):
        del net.params[name]

    def removeLayer(self, name):
        # todo: fix this stuff for weight sharing
        layer = self.layers[name]
        for paramName in layer.params:
            self.removeParam(paramName)
        del self.layers[name]

    def reshape(self):
        for layer in self.layers.itervalues():
            layer.reshape(self)

    def display(self):
        for layer in self.layers.itervalues():
            layer.display()
        for var in self.vars.itervalues():
            print 'Variable ', var.name
            print '  comp. shape: %s' % (var.size,)
        for par in self.params.itervalues():
            print 'Parameter ', par.name
            print '   data found: %s' % (par.size is not None)
            print '   data shape: %s' % (str(par.value.shape))

    def transpose(self):
        for layer in self.layers.itervalues():
            layer.transpose(self)

    def getParentTransforms(self, variableName, topLayerName=None):
        layerNames = self.layers.keys()
        if topLayerName:
            layerIndex = layerNames.index(topLayerName)
        else:
            layerIndex = len(self.layers) + 1
        transforms = OrderedDict()
        transforms[variableName] = CaffeTransform([1.,1.], [1.,1.], [1.,1.])
        for layerName in reversed(layerNames[0:layerIndex]):
            layer = self.layers[layerName]
            layerTfs = layer.getTransforms(self)
            for i, inputName in enumerate(layer.inputs):
                tfs = []
                if transforms.has_key(inputName):
                    tfs.append(transforms[inputName])
                for j, outputName in enumerate(layer.outputs):
                    if layerTfs[i][j] is None: continue
                    if transforms.has_key(outputName):
                        composed = composeTransforms(layerTfs[i][j], transforms[outputName])
                        tfs.append(composed)

                if len(tfs) > 0:
                    # should resolve conflicts, not simply pick the first tf
                    transforms[inputName] = tfs[0]
        return transforms
