# MatConvNet: CNNs for MATLAB

**MatConvNet** is a MATLAB toolbox implementing *Convolutional Neural
Networks* (CNNs) for computer vision applications. It is simple,
efficient, and can run and learn state-of-the-art CNNs. Several
example CNNs are included to classify and encode images.

**Citing.** If you use MatConvNet in your work, please cite:
"MatConvNet - Convolutional Neural Networks for MATLAB", A. Vedaldi
and K. Lenc, *Proc. of the ACM Int. Conf. on Multimedia*, 2015. <span
style="color:#428bca;"
onclick="toggle_visibility('bibentry');">[BibTex]</span>

<pre class="shy" id="bibentry">
    @inproceedings{vedaldi15matconvnet,
         author    = {A. Vedaldi and K. Lenc},
         title     = {MatConvNet -- Convolutional Neural Networks for MATLAB},
         book      = {Proceeding of the {ACM} Int. Conf. on Multimedia}
         year      = {2015},
    }
</pre>

> **New:** 1.0-beta16 adds VGG-Face as a [pretrained model](pretrained.md).
>
> **New:** Fully-Convolutional Networks (FCN) training and evaluation
> code is available
> [here](https://github.com/vlfeat/matconvnet-fcn).
>
> **New:** 1.0-beta15 adds a few new layers to DagNN to support the
> **Fully-Convolutonal Networks** (FCN) for image
> segmentation. Pretrained models are
> [also available here](pretrained.md). Batch normalization
> ([`vl_nnbnorm`](mfiles/vl_nnbnorm.md)) has also been improved adding
> features that will make it easier to remove the layer after training
> a model.
>
> **New:** 1.0-beta14 adds a new object-oriented
> [network wrapper `DagNN`](wrappers.md) supporting arbitrary network
> topologies. This release also adds GoogLeNet as a pre-trained model,
> new building blocks such as [`vl_nnconcat`](mfiles/vl_nnconcat.md),
> a rewritten loss function block [`vl_nnloss`](mfiles/vl_nnloss.md),
> better documentation, and bugfixes. A new **realtime demo** (see
> `examples/cnn_imagenet_camdemo.m`) using GoogLeNet, VGG-VD, or any
> other similar network.
>
> **New:** MatConvNet used in planetary science research by the
> University of Arizona (see the
> [NVIDIA blog post](http://devblogs.nvidia.com/parallelforall/deep-learning-image-understanding-planetary-science/)).

*   **Obtaining MatConvNet**
    - Tarball for [version 1.0-beta16](download/matconvnet-1.0-beta16.tar.gz)
    - [GIT repository](http://www.github.com/vlfeat/matconvnet.git)

*   **Documentation**
    - [PDF manual](matconvnet-manual.pdf)
    - [MATLAB functions](functions.md)
    - [FAQ](faq.md)

*   **Getting started**
    - [Quick start guide](quick.md)
    - [Installation instructions](install.md)
    - [Using pre-trained models](pretrained.md): VGG-VD, GoogLeNet, FCN, ...
    - [Training your own models](training.md)
    - [CNN wrappers: linear chains or DAGs](wrappers.md)
    - [Working with GPU accelerated code](gpu.md)
    - [Tutorial](http://www.robots.ox.ac.uk/~vgg/practicals/cnn/index.html),
      [slides](http://www.robots.ox.ac.uk/~vedaldi/assets/teach/2015/vedaldi15aims-bigdata-lecture-4-deep-learning-handout.pdf)

*   **Other information**
    - [Changes](about/#changes)
    - [Developing the library](developers.md)

