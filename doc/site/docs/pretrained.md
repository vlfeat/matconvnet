# Pretrained models

This section describes how pre-trained models can be downloaded and
used in MatConvNet.

> **Remark:** The following CNN models may have been *imported from
> other reference implementations* and are equivalent to the originals
> up to numerical precision. However, note that:
>
> 1.  Images need to be pre-processed (resized and cropped) before
>     being submitted to a CNN for evaluation. Even small differences
>     in the prepreocessing details can have a non-negligible effect
>     on the results.
>
> 2.  The example below shows how to evaluate a CNN, but does not
>     include data augmentation or encoding normalization as for
>     example provided by the
>     [VGG code](http://www.robots.ox.ac.uk/~vgg/research/deep_eval).
>     While this is easy to implement, it is not done automatically
>     here.
>
> 3.  These models are provided here for convenience, but please
>     credit the original authors.

## Download the pretrained models

-    VGG models from the
     [Very Deep Convolutional Networks for Large-Scale Visual Recognition](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)
     project.

     **Citation:** `Very Deep Convolutional Networks for Large-Scale
     Image Recognition', *Karen Simonyan and Andrew Zisserman,* arXiv
     technical report, 2014, ([paper](http://arxiv.org/abs/1409.1556/)).

     - [imagenet-vgg-verydeep-16](models/imagenet-vgg-verydeep-16.mat)
     - [imagenet-vgg-verydeep-19](models/imagenet-vgg-verydeep-19.mat)

-    VGG models from the
     [Return of the Devil](http://www.robots.ox.ac.uk/~vgg/research/deep_eval)
     paper (v1.0.1).

     **Citation:** `Return of the Devil in the Details: Delving Deep
     into Convolutional Networks', *Ken Chatfield, Karen Simonyan,
     Andrea Vedaldi, and Andrew Zisserman,* BMVC 2014
     ([BibTex and paper](http://www.robots.ox.ac.uk/~vgg/publications/2014/Chatfield14/)).

     - [imagenet-vgg-f](models/imagenet-vgg-f.mat)
     - [imagenet-vgg-m](models/imagenet-vgg-m.mat)
     - [imagenet-vgg-s](models/imagenet-vgg-s.mat)
     - [imagenet-vgg-m-2048](models/imagenet-vgg-m-2048.mat)
     - [imagenet-vgg-m-1024](models/imagenet-vgg-m-1024.mat)
     - [imagenet-vgg-m-128](models/imagenet-vgg-m-128.mat)

-    Berkeley
     [Caffe reference models](http://caffe.berkeleyvision.org/getting_pretrained_models.html)
     (version downloaded on September 2014).

     **Citation:** Please see [Caffe homepage](http://caffe.berkeleyvision.org).

     - [imagenet-caffe-ref](models/imagenet-caffe-ref.mat)
     - [imagenet-caffe-alex](models/imagenet-caffe-alex.mat)

This is a summary of the performance of these models on the ILSVRC
2012 validation data:


|               model|top-1 err.|top-5 err.|  images/s|
|--------------------|----------|----------|----------|
|           caffe-ref|      42.7|      19.8|     205.4|
|          caffe-alex|      42.9|      19.8|     274.8|
|               vgg-s|      36.9|      15.4|     312.2|
|               vgg-m|      37.5|      16.1|     382.8|
|               vgg-f|      41.5|      19.1|     638.0|
|     vgg-verydeep-19|      29.0|      10.1|      57.1|
|     vgg-verydeep-16|      28.8|      10.1|      68.3|

Note that these error rates are computed on a single centre-crop and
are therefore higher than what reported in some publications, where
multiple evaluations per image are combined.

The evaluation speed was measured on a 12-cores machine using a single
NVIDIA Titan Black GPU and MATLAB R2015a; performance varies hugely
depending on the network but also on how the data was preprocessed;
for example, `caffe-ref` and `caffe-alex` should be as fast as
`vgg-f`, but they are not since images were pre-processed in such a
way that MATLAB had to call `imresize` for each input image.

## Using the pretrained models

In order to run, say, `imagenet-vgg-s` on a test image, use:

    % setup MtConvNet in MATLAB
    run matlab/vl_setupnn

    % download a pre-trained CNN from the web
    urlwrite('http://www.vlfeat.org/sandbox-matconvnet/models/imagenet-vgg-f.mat', ...
      'imagenet-vgg-f.mat') ;
    net = load('imagenet-vgg-f.mat') ;

    % obtain and preprocess an image
    im = imread('peppers.png') ;
    im_ = single(im) ; % note: 255 range
    im_ = imresize(im_, net.normalization.imageSize(1:2)) ;
    im_ = im_ - net.normalization.averageImage ;

    % run the CNN
    res = vl_simplenn(net, im_) ;

    % show the classification result
    scores = squeeze(gather(res(end).x)) ;
    [bestScore, best] = max(scores) ;
    figure(1) ; clf ; imagesc(im) ;
    title(sprintf('%s (%d), score %.3f',...
    net.classes.description{best}, best, bestScore)) ;

`vl_simplenn` is a wrapper around MatConvNet core computational blocks
that implements a CNN with a simple linear structure (a chain of
layers). It is not needed to use the toolbox, but it simplifies common
examples such as the ones discussed here. See also

