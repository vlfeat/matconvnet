# Frequently-asked questions (FAQ)

## Running MatConvNet

### Can I use MatConvNet with CuDNN?

Yes, but CuDNN must be installed and linked to MatConvNet. See the
[installation instructions](install.md).

### How do I fix the error `Attempt to execute SCRIPT vl_nnconv as a function`?

Before the toolbox can be used, the
[MEX files](http://www.mathworks.com/support/tech-notes/1600/1605.html
) must be compiled. Make sure to follow the
[installation instructions](install.md). If you have done so and the
MEX files are still not recognized, check that the directory
`matlab/toolbox/mex` contains the missing files. If the files are
there, there may be a problem with the way MEX files have been
compiled.

### Why files such as `vl_nnconv.m` do not contain any code?

Functions such as `vl_nnconv`, `vl_nnpool`, `vl_nnbnorm` and many
others are implemented MEX files. In this case, M files such as
`vl_nnconv.m` contain only the function documentation. The code of the
function is actually found in `matlab/src/vl_nnconv.cu` (a CUDA/C++
source file) or similar.
