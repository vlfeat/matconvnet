# MatConvNet: CNNs for MATLAB

**MatConvNet** is a MATLAB toolbox implementing *Convolutional Neural
Networks* (CNNs) for computer vision applications. It is simple,
efficient, and can run and learn state-of-the-art CNNs. Several
example CNNs are included to classify and encode images.

> **New:** 1.0-beta14 adds a new object-oriented network wrapper,
> `DagNN`, supporting arbitrary network topologies. This release also
> adds GoogLeNet as a pre-trained model, new building blocks such as
> `vl_nnconcat`, a rewritten loss function block `vl_nnloss`,
> better documentation, and bugfixes.
>
> **New:** 1.0-beta13 adds a much faster version of batch
> normalization and contains several bugfixes and minor improvements.
>
> **New:** MatConvNet used in planetary science research by the
> University of Arizona (see the
> [NVIDIA blog post](http://devblogs.nvidia.com/parallelforall/deep-learning-image-understanding-planetary-science/)).
>
> **New:** 1.0-beta12 adds convolution transpose aka deconvolution
> (see [`vl_nnconvt`](mfiles/vl_nnconvt)).

*   **Obtaining MatConvNet**
    - Tarball for [version 1.0-beta14](download/matconvnet-1.0-beta14.tar.gz)
    - [GIT repository](http://www.github.com/vlfeat/matconvnet.git)

*   **Documentation**
    - [PDF manual](matconvnet-manual.pdf)
    - [MATLAB functions](functions.md)
    - [FAQ](faq.md)

*   **Getting started**
    - [Quick start guide](quick.md)
    - [Installation instructions](install.md)
    - [Using pre-trained models](pretrained.md)
    - [Training your own models](training.md)
    - [Working with GPU accelerated code](gpu.md)
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
