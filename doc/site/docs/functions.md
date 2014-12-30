# MATLAB API

MatConvNet comprises several MATLAB functions located in the
`matlab/` subdirectory. These are divided as follows:

- [Core functions](#core). These comprise the fundamental
  computational building blocks that can be recombined to implement a
  variety of CNNs, beyond the ones supported by the provided wrappers.
- [Simple CNN wrapper functions](#simplenn). These function implement a wrapper
  for CNNs with a simple layout (a chain).
- [Utility functions](#utility). Helper functions to initialize the
  toolbox and similar.

Note that there is no general training function as this depends on the
specific of the problem and dataset. Look at the examples to know how
to do this.

## Core functions

<a name="core"/>

- [`vl_nnconv`](mfiles/vl_nnconv.md) Linear convolution by a filter
  bank (and fully connected layer).
- [`vl_nndropout`](mfiles/vl_nndropout.md) Dropout.
- [`vl_nnloss`](mfiles/vl_nnloss.md) Classification log-loss.
- [`vl_nnnormalize`](mfiles/vl_nnnormalize.md) Channel-wise group normalization.
- [`vl_nnnoffset`](mfiles/vl_nnnoffset.md) Norm-dependent offset.
- [`vl_nnpool`](mfiles/vl_nnpool.md) Max and sum pooling.
- [`vl_nnrelu`](mfiles/vl_nnrelu.md) REctified Linear Unit.
- [`vl_nnsoftmax`](mfiles/vl_nnsoftmax.md) Channel-wise soft-max.
- [`vl_nnsoftmaxloss`](mfiles/vl_nnsoftmaxloss.md) Combination of soft-max and log-loss.

## Simple CNN functions

<a name="simplenn"/>

- [`vl_simplenn`](mfiles/vl_simplenn.md) A wrapper for a CNN with a
  simple (linear) layout
- [`vl_simplenn_diagnose`](mfiles/vl_simplenn_diagnose.md) Print diagnostics about the CNN
- [`vi_simplenn_display`](mfiles/vl_simplenn_display.md) Print information about the CNN architecture
- [`vl_simplenn_move`](mfiles/vl_simplenn_move.md) Move the CNN between CPU and GPU

## Utility functions

<a name="utility"/>

- [`vl_argparse`](mfiles/vl_argparse.md) A helper function to parse
  optional arugments.
- [`vl_compilenn`](mfiles/vl_compilenn.md) Compile the MEX fiels in the toolbox.
- [`vl_rootnn`](mfiles/vl_rootnn.md) Return the path to the MatConvNet toolbox installation.
- [`vl_setpunn`](mfiles/vl_setupnn.md) Setup MatConvNet for use in MATLAB.
