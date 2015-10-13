# Pretrained models

This section describes how pre-trained models can be downloaded and
used in MatConvNet. Using the pre-trained model is easy; just start
from the example code included in the [quickstart guide](quick.md).

[TOC]

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

## Face recognition

These models are trained for face classification and verification.

-   **VGG-Face**. The face classification and verification network from
    the
    [VGG project](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/).

    > Deep face recognition, O. M. Parkhi and A. Vedaldi and
    > A. Zisserman, Proceedings of the British Machine Vision
    > Conference (BMVC), 2015
    > ([paper](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf)).

    - [vgg-face](models/vgg-face.mat)

    See the script `examples/cnn_vgg_face.m` for an example of using
    VGG-Face in 'classifcation' mode. To use this for face
    verification, extract the 4K dimensional features by removing the
    last classification layer and normalize the resulting vector in L2
    norm.

## Semantic segmentation

These models are trained for semantic image segmentation using the
PASCAL VOC category definitions.

-   **Fully-Convolutional Networks** (FCN) training and evaluation
    code is available
    [here](https://github.com/vlfeat/matconvnet-fcn).

-   **BVLC FCN** (the original implementation) imported from the
    [Caffe version](https://github.com/BVLC/caffe/wiki/Model-Zoo)
    [*DagNN format*].

    > 'Fully Convolutional Models for Semantic Segmentation',
    *Jonathan Long, Evan Shelhamer and Trevor Darrell*, CVPR, 2015
    ([paper](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf)).

    - [pascal-fcn32s-dag](models/pascal-fcn32s-dag.mat)
    - [pascal-fcn16s-dag](models/pascal-fcn16s-dag.mat)
    - [pascal-fcn8s-dag](models/pascal-fcn8s-dag.mat)

    These networks are trained on the PASCAL VOC 2011 training and (in
    part) validation data, using Berekely's extended annotations
    ([SBD](http://www.cs.berkeley.edu/~bharath2/codes/SBD/download.html)).

    The performance measured on the PASCAL VOC 2011 validation data
    subset used in the revised version of the paper above (dubbed
    RV-VOC11):

    | Model   | Test data |Mean IOU | Mean pix. accuracy | Pixel accuracy |
    |---------|-----------|---------|--------------------|----------------|
    | FNC-32s | RV-VOC11  | 59.43   | 89.12              | 73.28          |
    | FNC-16s | RV-VOC11  | 62.35   | 90.02              | 75.74          |
    | FNC-8s  | RV-VOC11  | 62.69   | 90.33              | 75.86          |

-   **Torr Vision Group FCN-8s**. This is the FCN-8s subcomponent of the
    CRF-RNN network from the paper:

    > 'Conditional Random Fields as Recurrent Neural Networks' *Shuai
    > Zheng, Sadeep Jayasumana, Bernardino Romera-Paredes, Vibhav
    > Vineet, Zhizhong Su, Dalong Du, Chang Huang, and Philip
    > H. S. Torr*,
    > ICCV 2015 ([paper](http://www.robots.ox.ac.uk/~szheng/papers/CRFasRNN.pdf)).

    - [pascal-fcn8s-tvg-dag](models/pascal-fcn8s-tvg-dag.mat)

    These networks are trained on the PASCAL VOC 2011 training and (in
    part) validation data, using Berekely's extended annotations, as
    well as Microsoft COCO.

    While the CRF component is missing (it may come later to
    MatConvNet), this model still outperforms the FCN-8s network
    above, partially because it is trained with additional data from
    COCO. In the table below, the RV-VOC12 data is the subset of the
    PASCAL VOC 12 data as described in the 'Conditional Random Fields'
    paper:

    | Model      | Tes data  | mean IOU | mean pix. accuracy | pixel accuracy |
    |------------|-----------|----------|--------------------|----------------|
    | FNC-8s-TVG | RV-VOC12  | 69.85    | 92.94              | 78.80          |

    *TVG implementation note*: The model was obtained by first
    fine-tuning the plain FCN-32s network (without the CRF-RNN part)
    on COCO data, then building built an FCN-8s network with the
    learnt weights, and finally training the CRF-RNN network
    end-to-end using VOC 2012 training data only. The model available
    here is the FCN-8s part of this network (without CRF-RNN, while
    trained with 10 iterations CRF-RNN).

## ImageNet ILSVRC classification

These modesl are trained to perform classification in the ImageNet
ILSVRC challenge data.

-   **GoogLeNet** model imported from the
    [Princeton version](http://vision.princeton.edu/pvt/GoogLeNet/)
    [*DagNN format*].

    > `Going Deeper with Convolutions', *Christian Szegedy, Wei Liu,
    Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov,
    Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich*, CVPR, 2015
    ([paper](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf)).

    - [imagenet-googlenet-dag](models/imagenet-googlenet-dag.mat)

-   **VGG-VD** models from the
    [Very Deep Convolutional Networks for Large-Scale Visual Recognition](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) project.

    > `Very Deep Convolutional Networks for Large-Scale Image
    Recognition', *Karen Simonyan and Andrew Zisserman,* arXiv
    technical report, 2014,
    ([paper](http://arxiv.org/abs/1409.1556/)).

    - [imagenet-vgg-verydeep-16](models/imagenet-vgg-verydeep-16.mat)
    - [imagenet-vgg-verydeep-19](models/imagenet-vgg-verydeep-19.mat)

-   **VGG-S,M,F** models from the
    [Return of the Devil](http://www.robots.ox.ac.uk/~vgg/research/deep_eval)
    paper (v1.0.1).

    > `Return of the Devil in the Details: Delving Deep into
    Convolutional Networks', *Ken Chatfield, Karen Simonyan, Andrea
    Vedaldi, and Andrew Zisserman,* BMVC 2014
    ([BibTex and paper](http://www.robots.ox.ac.uk/~vgg/publications/2014/Chatfield14/)).

    - [imagenet-vgg-f](models/imagenet-vgg-f.mat)
    - [imagenet-vgg-m](models/imagenet-vgg-m.mat)
    - [imagenet-vgg-s](models/imagenet-vgg-s.mat)
    - [imagenet-vgg-m-2048](models/imagenet-vgg-m-2048.mat)
    - [imagenet-vgg-m-1024](models/imagenet-vgg-m-1024.mat)
    - [imagenet-vgg-m-128](models/imagenet-vgg-m-128.mat)

-   **Berkeley**
    [Caffe reference models](http://caffe.berkeleyvision.org/getting_pretrained_models.html)
    (version downloaded on September 2014).

    > Citation: please see the [Caffe homepage](http://caffe.berkeleyvision.org).

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
