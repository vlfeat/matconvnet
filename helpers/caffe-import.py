#! /bin/usr/python

import caffe_pb2

param_file='../data/vgg/CNN_F/param.proto'
model_fiel='../data/vgg/CNN_F/model'

f = open(param_file)
caffe_pb2.ParseFromString(f.read())
