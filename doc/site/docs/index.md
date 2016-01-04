# MatConvNet: CNNs for MATLAB

<div class="row" style="white-space: nowrap;">
<div class="col-sm-3">
<a href="download/matconvnet-1.0-beta17.tar.gz">
<div class="menuicon"><span class="fa fa-download fa-2x"></span></div>
Download</a>
</div>
<div class="col-sm-3">
<a href="http://www.github.com/vlfeat/matconvnet.git">
<div class="menuicon"><span class="fa fa-github fa-2x"></span></div>
Code &amp; issues</a>
</div>
<div class="col-sm-3">
<a href="pretrained/">
<div class="menuicon"><span class="fa fa-cubes fa-2x"></span></div>
Pre-trained models
</a>
</div>
<div class="col-sm-3">
<a href="https://groups.google.com/d/forum/matconvnet">
<div class="menuicon"><span class="fa fa-comments fa-2x"></span></div>
Discussion forum
</a>
</div>
</div>

**MatConvNet** is a MATLAB toolbox implementing *Convolutional Neural
Networks* (CNNs) for computer vision applications. It is simple,
efficient, and can run and learn state-of-the-art CNNs. Many
pre-trained CNNs for image classification, segmentation, face
recognition, and text detection are available.


> **New:** [1.0-beta17](about.md#changes) tidies up the library in
> many ways. It also improves how batch normalization is
> handled. Please check carefully the [changes](about.md#changes) to
> see if any of this might impact your workflow. In particular, check
> out the [`vl_simplenn_tidy`](mfiles/simplenn/vl_simplenn_tidy.md)
> function to bring old models up to date. We also have opened a new
> [discussion group](https://groups.google.com/d/forum/matconvnet) for
> MatConvNet users (please use this group for discussions and GitHub
> for reporting bugs or similar). cuDNN v4 is also supported (although
> not all v4 features are used yet). Several pre-trained ImageNet models
> trained with MatConvNet using the supplied *example code* are now
> available.
>
> **New:** 1.0-beta16 adds VGG-Face as a [pretrained model](pretrained.md).
>
> **New:** Fully-Convolutional Networks (FCN) training and evaluation
> code is available
> [here](https://github.com/vlfeat/matconvnet-fcn).

## Obtaining MatConvNet
- <span class="fa fa-file-archive-o"></span>&nbsp;Tarball for [version 1.0-beta17](download/matconvnet-1.0-beta17.tar.gz); [older versions](download/) (<span class="fa fa-apple"/> <span class="fa fa-windows"/> <span class="fa fa-linux"/>)
- <span class="fa fa-github"></span>&nbsp;[GIT repository](http://www.github.com/vlfeat/matconvnet.git)
- <span class="fa fa-pencil-square-o"></span>&nbsp;<a href="javascript:void(0);"
  onclick="toggle_visibility('citation');">Citation</a>
  <div class="shy" id="citation">
  "MatConvNet - Convolutional Neural Networks for MATLAB", A. Vedaldi
  and K. Lenc, *Proc. of the ACM Int. Conf. on Multimedia*, 2015.
  <pre>
  @inproceedings{vedaldi15matconvnet,
      author    = {A. Vedaldi and K. Lenc},
      title     = {MatConvNet -- Convolutional Neural Networks for MATLAB},
      book      = {Proceeding of the {ACM} Int. Conf. on Multimedia}
      year      = {2015},
  }</pre>
  </div>

## Documentation
- <span class="fa fa-book"></span> [Manual](matconvnet-manual.pdf) (PDF)
- <span class="fa fa-puzzle-piece"></span> [MATLAB functions](functions.md)
- <span class="fa fa-question-circle"></span> [FAQ](faq.md)
- <span class="fa fa-comments"></span> [Discussion group](https://groups.google.com/d/forum/matconvnet)

## Getting started
- [Quick start guide](quick.md)
- [Installation instructions](install.md)
- [Using pre-trained models](pretrained.md): VGG-VD, GoogLeNet, FCN, ...
- [Training your own models](training.md)
- [CNN wrappers: linear chains or DAGs](wrappers.md)
- [Working with GPU accelerated code](gpu.md)
- [Tutorial](http://www.robots.ox.ac.uk/~vgg/practicals/cnn/index.html),
  [slides](http://www.robots.ox.ac.uk/~vedaldi/assets/teach/2015/vedaldi15aims-bigdata-lecture-4-deep-learning-handout.pdf)

## Use cases
- The computer vision course at MIT is using MatConvNet for their [final project](http://6.869.csail.mit.edu/fa15/project.html)
- Deep Learning for Computer Vision with MATLAB and cuDNN ([NVIDIA...](http://devblogs.nvidia.com/parallelforall/deep-learning-for-computer-vision-with-matlab-and-cudnn/))
- Planetary science research by the  University of Arizona ([NVIDIA...](http://devblogs.nvidia.com/parallelforall/deep-learning-image-understanding-planetary-science/))

## Other information
- [Changes](about/#changes)
- [Developing the library](developers.md)

