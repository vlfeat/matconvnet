# Frequently-asked questions (FAQ)

## Running MatConvNet

<a name=mex></a>
### How do I fix the error `Attempt to execute SCRIPT vl_nnconv as a function`?

Before the toolbox can be used, the [MEX files](http://www.mathworks.com/support/tech-notes/1600/1605.html ) must be compiled. Make sure to follow the [installation instructions](install.md). If you have done so and the MEX files are still not recognized, check that the directory `matlab/toolbox/mex` contains the missing files. If the files are there, there may be a problem with the way MEX files have been compiled.

### Why files such as `vl_nnconv.m` do not contain any code?

These functions are [MEX files](#mex).

### Do I need a specific C++ compiler version?

Officially, different MATLAB versions support only specific versions of each compiler (Xcode, GCC, and Visual Studio). If needed, it is usually possible to install an old compiler version *alongside* any newer version you may have.

### Do I need a specific version of the CUDA devkit?

Officially, MathWorks supports a specific version of the CUDA devkit with each MATLAB version (see [here](install.md#gpu)). However, in practice we normally use the most recent version of CUDA (and cuDNN) available from NVIDIA without problems (see [here](install.md#nvcc)). This will require using the `nvcc` compilation method or to customize the mex [configuration files](#hack).

### Can I use MatConvNet with CuDNN?

Yes, and this is the recommended way of running MatConvNet on NVIDIA GPUs. However, you need to install cuDNN and link it to MatConvNet. See the [installation instructions](install.md#cudnn) to know how.

<a name=hack></a>
### How do I customize the `mex` configuration files?

Stubborn compilation problems can usually be solved by tweaking the configuration of the `mex` compiler. This is done by editing the mex configuration files. In order do so, start by creating a copy of the relevant files, apply your changes, and use the `MexConfig` and `MexCudaConfig` options of `vl_compilenn` to use them instead of the default ones.

There are two flavour of such files relevant to MatConvNet:

*   The mex configuration for **C++** is automatically copied to the user's MATLAB preference directory by running `mex -setup`. The preference directory can be discovered by using the `prefdir()` command in MATLAB and the file has a name of the type `mex_C++_glnxa64.xml`, `mex_C++_maci64.xml`,  `mex_C++_win64`, or similar.

    It is safe to edit this file, but it will be overwritten upon calling `mex -setup`, so consider creating a copy instead and the use the `MexConfig` option of `vl_compilenn` to use it.

*   The mex configuration for **CUDA**. This is *not* copied to the user's preference directory; its location can be found by using the MATLAB command `fullfile(toolboxdir('distcomp'), 'gpu', 'extern', 'src', 'mex', computer('arch'))` and the file name will be of the type `nvcc_clang++.xml`, `nvcc_g++.xml` or `nvcc_msvcpp2015.xml`. Copy it to a custom directory (for example MatConvNet folder or the user's preference directory `prefdir()`), edit it, and then use the `MexCudaConfig` option to instruct `vl_compilenn` to use it.

<a name=compiler></a>
### I am getting the error `No supported compiler or SDK was found`. What can I do?

This occurs when MATLAB cannot identify a compatible compiler version. Usually, this is most likely to occur when compiling CUDA code using the `mex` method, as both constraints imposed by MATLAB and the CUDA compiler NVCC are combined. There are three solutions:

1. Try to compile CUDA code using the `nvcc` method instead of the `mex` method. This skips MATLAB and uses NVCC directly to compile CUDA code relaxing many constraints.

2. Install the exact compiler and CUDA requirements imposed by the specific MATLAB (note that each MATLAB MATLAB has an officially-supported CUDA version a corresponding compiler version). For macOS see also [here](#macos).

3. Hack the `mex` configuration files to relax some constraints as explained [above](#hack). Some of the issues that can be addressed by this method are further discussed in the other points of the FAQ.

### I am getting the `error: 'constexpr' does not name a type` when compiling MatConvNet. What can I do?

You are likely compiling MatConvNet CUDA code using the `mex` method and an old MATLAB version. The `mex` configuration file contained an odd bug for which both the options `-std=c++11` and `-ansi` would be specified. Either:

1. Use the `nvcc` method to compile MatConvNet or
2. Hack the mex CUDA configuration file as explained [above](#hack) and remove any occurrence of the string `-ansi`.

### Linux: Why do I get compilation error `error: unrecognized command line option "-std=c++11"`?

Use a GCC that supports C++11; yours is likely too old ([<4.6](https://gcc.gnu.org/projects/cxx-status.html#cxx11)).

<a name=macos></a>
### macOS: I have Xcode installed, but I still cannot compile MatConvNet. What can I do?

There are a few common sources of issues:

1.   `mex -setup` fails because Xcode is [not compatible](#compiler) with your MATLAB version. The solution is to [install an old version of Xcode](#oldxcode).
2.   `mex -setup` cannot find a compatible SDK. A solution is suggested [here](#oldsdk).

<a name=oldxcode></a>
### macOS: How can I use an old version of Xcode?

MATLAB can be picky about the supported version of Xcode, and further restrictions may apply when using GPU code (see below). If `mex -setup` complains that no suitable Xcode version can be found, the safest option is to download an old version of Xcode (from [developer.apple.com](http://developer.apple.com/download/more), requires a free developer account) and install it in a location such as `/Applications/Xcode7.3.1.app`.

You can then:

1. Start MATLAB form the command line after setting the `DEVELOPER_DIR` environment variable, as in
   ```
   export DEVELOPER_DIR=/Applications/Xcode7.3.1.app/Contents/Developer
   /Applications/MATLAB_R2016a.app/bin/matlab
   ```
   where paths need to be adjusted to your situation. The advantage of this method is that it does not require changing your default Xcode selection.

2. Use `xcode-select` to activate the old version of Xcode globally and then `xcode-select --install` to install the corresponding command line tools; for example:
   ```
   sudo xcode-select --switch /Applications/Xcode7.3.1.app/Contents/Developer
   sudo xcode-select --install
   ```

After this, `mex -setup` should be able to find and use the old compiler.

<a name=oldsdk></a>
### macOS: mex is complaining about missing SDKs. What can I do?

If `mex -setup` still fails, the cause is likely your combination of Xcode and macOS versions: MATLAB expects an old version of the **macOS SDK** to be installed in Xcode, but this does not show up as macOS is more recent than that.

Sometimes this issue is patched by Mathworks; alternatively, you can [tweak](https://uk.mathworks.com/matlabcentral/answers/243868-mex-can-t-find-compiler-after-xcode-7-update-r2015b) the `mex` configuration file as explained [above](#hack). Look for the `<ISYSROOT>` and `<SDKVER>` sections of the file and add lines such as `<dirExists name="$$/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.11.sdk" />` as needed for your SDK version. Note that these are overwritten if you call `mex -setup` again, so consider creating copies. You will need to perform similar changes to the corresponding CUDA configuration files.
