# MatConvNet: CNNs for MATLAB

**MatConvNet** is a MATLAB toolbox implementing *Convolutional Neural
Networks* (CNNs) for computer vision applications. It is simple,
efficient, and can run and learn state-of-the-art CNNs. Several
example CNNs are included to classify and encode images. Please visit
the [homepage](http://www.vlfeat.org/matconvnet) to know more.

In case of compilation issues, please read first the
[Installation](http://www.vlfeat.org/matconvnet/install/) and
[FAQ](http://www.vlfeat.org/matconvnet/faq/) section before creating an GitHub
issue. For general inquiries regarding network design and training
related questions, please use the
[Discussion forum](https://groups.google.com/d/forum/matconvnet).

## What's New
This fork adds various changes to the original repo:
- [2-D RNN](https://arxiv.org/abs/1509.00552) modules:
  + 2-D RNN
  + 2-D [LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
  + 2-D GRU
- Online hard example mining ([OHEM](https://arxiv.org/abs/1604.03540)) SoftMax ([reference](https://github.com/itijyou/ademxapp/blob/master/util/symbol/layer.py))
- "[Interlaced](https://github.com/TuSimple/TuSimple-DUC)" conv (for upsampling)
- Various metrics:
  + [semantic segmentation](https://github.com/vlfeat/matconvnet-fcn): pixel Acc, mean Acc, mean IU
  + [Rand index](https://www.frontiersin.org/articles/10.3389/fnana.2015.00142/full) metric
- [UnPooling](https://github.com/peiyunh/matconvnet)
- [Reverse grad](https://arxiv.org/abs/1409.7495)
- Euclidean loss
- Adds `requiresGrad` in `DagNN.params` (similar to `requires_grad` in [PyTorch](http://pytorch.org/))
- `cnn_train_dag.m` (with new features)
