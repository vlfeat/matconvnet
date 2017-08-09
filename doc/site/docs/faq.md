# Frequently-asked questions (FAQ)

## Running MatConvNet

### Do I need a specific version of the CUDA devkit?

Officially, MathWorks supports a specific version of the CUDA devkit
with each MATLAB version (see [here](install.md#gpu)). However, in
practice we normally use the most recent version of CUDA (and cuDNN)
available from NVIDIA without problems (see
[here](install.md#nvcc)).

### Can I use MatConvNet with CuDNN?

Yes, and this is the recommended way of running MatConvNet on NVIDIA
GPUs. However, you need to install cuDNN and link it to
MatConvNet. See the [installation instructions](install.md#cudnn) to
know how.

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

### Why do I get compilation error `error: unrecognized command line option "-std=c++11"` on a Linux machine?

Use a GCC that supports C++11; yours is likely too old
([<4.6](https://gcc.gnu.org/projects/cxx-status.html#cxx11)).

<a name=macos></a>

### I have Xcode on a Mac but compiling MatConvNet still gives me errors. Why?

MATLAB can be picky about the supported version of Xcode, and further restrictions may apply
when using GPU code (see below). If `mex -setup` complains that no suitable
Xcode version can be found, the safest option is to download an old version of
 Xcode (from [developer.apple.com](http://developer.apple.com/download/more), requires a free Developeraccount) and install
it in a location such as `/Applications/Xcode7.3.1.app`.

You can then:

1. Use `xcode-select` to activate the old version of Xcode and then `xcode-select --install` globally andinstall the corresponding command line tools respectively; for example:
   ```
   sudo xcode-select --switch /Applications/Xcode7.3.1.app/Contents/Developer/
   sudo xcode-select --install
   ```
2. Alternatively, start MATLAB form the command line after setting the `DEVELOPER_DIR` environment variable, as in
   ```
   DEVELOPER_DIR=/Applications/Xcode7.3.1.app/Contents/Developer \
      /Applications/MATLAB_R2016a.app/bin/matlab
   ```
   where paths need to be adjusted to your situation. The advantage of this method is that it does not require changing your default Xcode selection.

After this, `mex -setup` should be able to find and use the old compiler.

If `mex -setup` still fails, the cause is likely your combination of Xcode and macOS versions: MATLAB expects an old version of the macOS SDK to be installed in Xcode, but this does not show up as macOS is more recent than that.

Sometimes this issue is patched by Mathworks; alternatively, you can edit the [configuration
 files](https://uk.mathworks.com/matlabcentral/answers/243868-mex-can-t-find-compiler-after-xcode-7-update-r2015b) `/Users/<home>/.matlab/R20???/mex_{C,C++}_maci64.xml`. Look for the `<ISYSROOT>` and `<SDKVER>` sections of the file and add lines such as `<dirExists name="$$/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.11.sdk" />` as needed for your SDK version. Note that these are overwritten if you call `mex -setup` again (alternatively you can
 change the corresponding files in MATLAB in `/Applications/MATLAB_R20???.app/bin/maci64/mexopts` for good).
