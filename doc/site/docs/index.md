# MatConvNet: CNNs for MATLAB

**MatConvNet** is a MATLAB toolbox implementing *Convolutional Neural
Networks* (CNNs) for computer vision applications. It is simple,
efficient, and can run and learn state-of-the-art CNNs. Several
example CNNs are included to classify and encode images.

> **New:** We have added [cuDNN](install.md#cudnn) support in
> 1.0-beta9. This may have significant benefit in speed and memory
> consupmtion.
>
> **New:** There is a new introductory <a
> href='http://www.robots.ox.ac.uk/~vgg/practicals/cnn/index.html'>VGG
> practical</a> on CNNs.

*   **Obtaining MatConvNet**
    - Tarball for [version 1.0-beta9](download/matconvnet-1.0-beta9.tar.gz)
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
