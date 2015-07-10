# MatConvNet: CNNs for MATLAB

**MatConvNet** is a MATLAB toolbox implementing *Convolutional Neural
Networks* (CNNs) for computer vision applications. It is simple,
efficient, and can run and learn state-of-the-art CNNs. Several
example CNNs are included to classify and encode images.

> **New:** 1.0-beta13 adds a much faster version of batch
> normalization and contains several bugfixes and minor improvements.
>
> **New:** MatConvNet used in planetary science research by the
> University of Arizona (see the
> [NVIDIA blog post](http://devblogs.nvidia.com/parallelforall/deep-learning-image-understanding-planetary-science/)).
>
> **New:** 1.0-beta12 adds convolution transpose aka deconvolution
> (see [`vl_nnconvt`](mfiles/vl_nnconvt)).
>
> **New:** 1.0-beta11 adds the following blocks: batch normalization
> (see [`vl_nnbnorm`](mfiles/vl_nnbnorm)), spatial normalization
> ([`vl_nnspnorm`](mfiles/vl_nnspnorm), p-distance
> ([`vl_nnpdist`](mfiles/vl_nnpdist)), sigmoid
> ([`vl_nnsigmoid`](mfiles/vl_nnsigmoid)). Extends the example trainig
> code `cnn_train` to support multiple GPUs. Improves the ImageNet and
> CIFAR examples, including batch normalization for ImageNet and the
> Network in Network model for CIFAR. See also
> [Changes](about/#changes) for compatibility considerations. A faster
> and more memory efficient version of batch normalization will arrive
> soon.

*   **Obtaining MatConvNet**
    - Tarball for [version 1.0-beta13](download/matconvnet-1.0-beta13.tar.gz)
    - [GIT repository](http://www.github.com/vlfeat/matconvnet.git)

*   **Documentation**
    - [PDF manual](matconvnet-manual.pdf)
    - [MATLAB functions](functions.md)

*   **Getting started**
    - [Installation instructions](install)
    - [Using pre-trained models](pretrained)
    - [Training your own models](training)
    - [Working with GPU accelerated code](gpu)
    - [Tutorial](http://www.robots.ox.ac.uk/~vgg/practicals/cnn/index.html),
      [slides](http://www.robots.ox.ac.uk/~vedaldi/assets/teach/2015/vedaldi15aims-bigdata-lecture-4-deep-learning-handout.pdf)

*   **Other information**
    - [Changes](about/#changes)
    - [Developing the library](developers.md)

**Citing.** If you use MatConvNet in your work, please cite:
"MatConvNet - Convolutional Neural Networks for MATLAB", A. Vedaldi
and K. Lenc, arXiv:1412.4564, 2014.

    @article{arXiv:1412.4564,
         author    = {A. Vedaldi and K. Lenc},
         title     = {MatConvNet -- Convolutional Neural Networks for MATLAB},
         journal   = {CoRR},
         volume    = {abs/1412.4564},
         year      = {2014},
    }
