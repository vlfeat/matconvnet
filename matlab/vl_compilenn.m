function vl_compilenn(varargin)
%VL_COMPILENN Compile the MatConvNet toolbox.
%   The `vl_compilenn()` function compiles the MEX files in the
%   MatConvNet toolbox. See below for the requirements for compiling
%   CPU and GPU code, respectively.
%
%   `vl_compilenn('OPTION', ARG, ...)` accepts the following options:
%
%   `EnableGpu`:: `false`
%      Set to true in order to enable GPU support.
%
%   `Verbose`:: 0
%      Set the verbosity level (0, 1 or 2).
%
%   `Continue`:: false
%      Avoid recreating a file if it was already compiled. This uses
%      a crude form of dependency checking, so it may occasionally be
%      necessary to rebuild MatConvNet without this option.
%
%   `Debug`:: `false`
%      Set to true to compile the binaries with debugging
%      information.
%
%   `CudaMethod`:: Linux & Mac OS X: `mex`; Windows: `nvcc`
%      Choose the method used to compile the CUDA code. There are two
%      methods:
%
%      * The **`mex`** method uses the MATLAB MEXCUDA command. This
%        is, in principle, the preferred method as it uses the
%        MATLAB-sanctioned compiler options.
%
%      * The **`nvcc`** method calls the NVIDIA CUDA compiler `nvcc`
%        directly to compile CUDA source code into object files.
%
%        This method allows to use a CUDA toolkit version that is not
%        the one that officially supported by a particular MATALB
%        version (see below). It is also the default method for
%        compilation under Windows and with CuDNN.
%
%   `CudaRoot`:: guessed automatically
%      This option specifies the path to the CUDA toolkit to use for
%      compilation.
%
%   `EnableImreadJpeg`:: `true`
%      Set this option to `true` to compile `vl_imreadjpeg`.
%
%   `EnableDouble`:: `true`
%      Set this option to `true` to compile the support for DOUBLE
%      data types.
%
%   `ImageLibrary`:: `libjpeg` (Linux), `gdiplus` (Windows), `quartz` (Mac)
%      The image library to use for `vl_impreadjpeg`.
%
%   `ImageLibraryCompileFlags`:: platform dependent
%      A cell-array of additional flags to use when compiling
%      `vl_imreadjpeg`.
%
%   `ImageLibraryLinkFlags`:: platform dependent
%      A cell-array of additional flags to use when linking
%      `vl_imreadjpeg`.
%
%   `EnableCudnn`:: `false`
%      Set to `true` to compile CuDNN support. See CuDNN
%      documentation for the Hardware/CUDA version requirements.
%
%   `CudnnRoot`:: `'local/'`
%      Directory containing the unpacked binaries and header files of
%      the CuDNN library.
%
%   `MexConfig`:: none
%      Use this option to specify a custom `.xml` configuration file
%      fot the `mex` compiler.
%
%   `MexCudaConfig`:: none
%      Use this option to specify a custom `.xml` configuration file
%      fot the `mexcuda` compiler.
%
%   `preCompileFn`:: none
%      Applies a custom modifier function just before compilation
%      to modify various compilation options. The
%      function's signature is:
%      [opts, mex_src, lib_src, flags] = f(opts, mex_src, lib_src, flags) ;
%      where the arguments are a struct with the present options, a list of
%      MEX files, a list of LIB files, and compilation flags, respectively.
%
%   ## Compiling the CPU code
%
%   By default, the `EnableGpu` option is switched to off, such that
%   the GPU code support is not compiled in.
%
%   Generally, you only need a 64bit C/C++ compiler (usually Xcode, GCC or
%   Visual Studio for Mac, Linux, and Windows respectively). The
%   compiler can be setup in MATLAB using the
%
%      mex -setup
%
%   command.
%
%   ## Compiling the GPU code
%
%   In order to compile the GPU code, set the `EnableGpu` option to
%   `true`. For this to work you will need:
%
%   * To satisfy all the requirements to compile the CPU code (see
%     above).
%
%   * A NVIDIA GPU with at least *compute capability 2.0*.
%
%   * The *MATALB Parallel Computing Toolbox*. This can be purchased
%     from Mathworks (type `ver` in MATLAB to see if this toolbox is
%     already comprised in your MATLAB installation; it often is).
%
%   * A copy of the *CUDA Devkit*, which can be downloaded for free
%     from NVIDIA. Note that each MATLAB version requires a
%     particular CUDA Devkit version:
%
%     | MATLAB version | Release | CUDA Devkit  |
%     |----------------|---------|--------------|
%     | 9.2            | 2017a   | 8.0          |
%     | 9.1            | 2016b   | 7.5          |
%     | 9.0            | 2016a   | 7.5          |
%     | 8.6            | 2015b   | 7.0          |
%
%     Different versions of CUDA may work using the hack described
%     above (i.e. setting the `CudaMethod` to `nvcc`).
%
%   The following configurations or anything more recent (subject to
%   versionconstraints between MATLAB, CUDA, and the compiler) should
%   work:
%
%   * Windows 10 x64, MATLAB R2015b, Visual C++ 2015, CUDA
%     Toolkit 8.0. Visual C++ 2013 and lower is not supported due to lack
%     C++11 support.
%   * macOS X 10.12, MATLAB R2016a, Xcode 7.3.1, CUDA
%     Toolkit 7.5-8.0.
%   * GNU/Linux, MATALB R2015b, gcc/g++ 4.8.5+, CUDA Toolkit 7.5-8.0.
%
%   Many older versions of these components are also likely to
%   work.
%
%   Compilation on Windows with MinGW compiler (the default mex compiler in
%   Matlab) is not supported. For Windows, please reconfigure mex to use
%   Visual Studio C/C++ compiler.
%   Furthermore your GPU card must have ComputeCapability >= 2.0 (see
%   output of `gpuDevice()`) in order to be able to run the GPU code.
%   To change the compute capabilities, for `mex` `CudaMethod` edit
%   the particular config file.  For the 'nvcc' method, compute
%   capability is guessed based on the GPUDEVICE output. You can
%   override it by setting the 'CudaArch' parameter (e.g. in case of
%   multiple GPUs with various architectures).
%
%   See also: [Compliling MatConvNet](../install.md#compiling),
%   [Compiling MEX files containing CUDA
%   code](http://mathworks.com/help/distcomp/run-mex-functions-containing-cuda-code.html),
%   `vl_setup()`, `vl_imreadjpeg()`.

% Copyright (C) 2014-17 Karel Lenc and Andrea Vedaldi.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

% Get MatConvNet root directory
root = fileparts(fileparts(mfilename('fullpath'))) ;
addpath(fullfile(root, 'matlab')) ;

% --------------------------------------------------------------------
%                                                        Parse options
% --------------------------------------------------------------------

opts.continue         = false;
opts.enableGpu        = false;
opts.enableImreadJpeg = true;
opts.enableCudnn      = false;
opts.enableDouble     = true;
opts.imageLibrary = [] ;
opts.imageLibraryCompileFlags = {} ;
opts.imageLibraryLinkFlags = [] ;
opts.verbose          = 0;
opts.debug            = false;
opts.cudaMethod       = [] ;
opts.cudaRoot         = [] ;
opts.cudaArch         = [] ;
opts.defCudaArch      = [...
  '-gencode=arch=compute_20,code=\"sm_20,compute_20\" '...
  '-gencode=arch=compute_30,code=\"sm_30,compute_30\"'];
opts.mexConfig        = '' ;
opts.mexCudaConfig    = '' ;
opts.cudnnRoot        = 'local/cudnn' ;
opts.preCompileFn       = [] ;
opts = vl_argparse(opts, varargin);

% --------------------------------------------------------------------
%                                                     Files to compile
% --------------------------------------------------------------------

arch = computer('arch') ;
check_compability(arch);
if isempty(opts.imageLibrary)
  switch arch
    case 'glnxa64', opts.imageLibrary = 'libjpeg' ;
    case 'maci64', opts.imageLibrary = 'quartz' ;
    case 'win64', opts.imageLibrary = 'gdiplus' ;
  end
end
if isempty(opts.imageLibraryLinkFlags)
  switch opts.imageLibrary
    case 'libjpeg', opts.imageLibraryLinkFlags = {'-ljpeg'} ;
    case 'quartz', opts.imageLibraryLinkFlags = {'-framework Cocoa -framework ImageIO'} ;
    case 'gdiplus', opts.imageLibraryLinkFlags = {'gdiplus.lib'} ;
  end
end

lib_src = {} ;
mex_src = {} ;

% Files that are compiled as CPP or CU depending on whether GPU support
% is enabled.
if opts.enableGpu, ext = 'cu' ; else ext='cpp' ; end
lib_src{end+1} = fullfile(root,'matlab','src','bits',['data.' ext]) ;
lib_src{end+1} = fullfile(root,'matlab','src','bits',['datamex.' ext]) ;
lib_src{end+1} = fullfile(root,'matlab','src','bits',['nnconv.' ext]) ;
lib_src{end+1} = fullfile(root,'matlab','src','bits',['nnfullyconnected.' ext]) ;
lib_src{end+1} = fullfile(root,'matlab','src','bits',['nnsubsample.' ext]) ;
lib_src{end+1} = fullfile(root,'matlab','src','bits',['nnpooling.' ext]) ;
lib_src{end+1} = fullfile(root,'matlab','src','bits',['nnnormalize.' ext]) ;
lib_src{end+1} = fullfile(root,'matlab','src','bits',['nnnormalizelp.' ext]) ;
lib_src{end+1} = fullfile(root,'matlab','src','bits',['nnbnorm.' ext]) ;
lib_src{end+1} = fullfile(root,'matlab','src','bits',['nnbias.' ext]) ;
lib_src{end+1} = fullfile(root,'matlab','src','bits',['nnbilinearsampler.' ext]) ;
lib_src{end+1} = fullfile(root,'matlab','src','bits',['nnroipooling.' ext]) ;
mex_src{end+1} = fullfile(root,'matlab','src',['vl_nnconv.' ext]) ;
mex_src{end+1} = fullfile(root,'matlab','src',['vl_nnconvt.' ext]) ;
mex_src{end+1} = fullfile(root,'matlab','src',['vl_nnpool.' ext]) ;
mex_src{end+1} = fullfile(root,'matlab','src',['vl_nnnormalize.' ext]) ;
mex_src{end+1} = fullfile(root,'matlab','src',['vl_nnnormalizelp.' ext]) ;
mex_src{end+1} = fullfile(root,'matlab','src',['vl_nnbnorm.' ext]) ;
mex_src{end+1} = fullfile(root,'matlab','src',['vl_nnbilinearsampler.' ext]) ;
mex_src{end+1} = fullfile(root,'matlab','src',['vl_nnroipool.' ext]) ;
mex_src{end+1} = fullfile(root,'matlab','src',['vl_taccummex.' ext]) ;
switch arch
  case {'glnxa64','maci64'}
    % not yet supported in windows
    mex_src{end+1} = fullfile(root,'matlab','src',['vl_tmove.' ext]) ;
end

% CPU-specific files
lib_src{end+1} = fullfile(root,'matlab','src','bits','impl','im2row_cpu.cpp') ;
lib_src{end+1} = fullfile(root,'matlab','src','bits','impl','copy_cpu.cpp') ;
lib_src{end+1} = fullfile(root,'matlab','src','bits','impl','tinythread.cpp') ;
lib_src{end+1} = fullfile(root,'matlab','src','bits','imread.cpp') ;

% GPU-specific files
if opts.enableGpu
  lib_src{end+1} = fullfile(root,'matlab','src','bits','impl','im2row_gpu.cu') ;
  lib_src{end+1} = fullfile(root,'matlab','src','bits','impl','copy_gpu.cu') ;
  lib_src{end+1} = fullfile(root,'matlab','src','bits','datacu.cu') ;
  mex_src{end+1} = fullfile(root,'matlab','src','vl_cudatool.cu') ;
end

% cuDNN-specific files
if opts.enableCudnn
end

% Other files
if opts.enableImreadJpeg
  mex_src{end+1} = fullfile(root,'matlab','src', ['vl_imreadjpeg.' ext]) ;
  mex_src{end+1} = fullfile(root,'matlab','src', ['vl_imreadjpeg_old.' ext]) ;
  lib_src{end+1} = fullfile(root,'matlab','src', 'bits', 'impl', ['imread_' opts.imageLibrary '.cpp']) ;
end

% --------------------------------------------------------------------
%                                                   Setup CUDA toolkit
% --------------------------------------------------------------------

if opts.enableGpu
  opts.verbose && fprintf('%s: * CUDA configuration *\n', mfilename) ;

  % Find the CUDA Devkit
  if isempty(opts.cudaRoot), opts.cudaRoot = search_cuda_devkit(opts) ; end
  opts.verbose && fprintf('%s:\tCUDA: using CUDA Devkit ''%s''.\n', ...
                          mfilename, opts.cudaRoot) ;

  opts.nvccPath = fullfile(opts.cudaRoot, 'bin', 'nvcc') ;
  switch arch
    case 'win64', opts.cudaLibDir = fullfile(opts.cudaRoot, 'lib', 'x64') ;
    case 'maci64', opts.cudaLibDir = fullfile(opts.cudaRoot, 'lib') ;
    case 'glnxa64', opts.cudaLibDir = fullfile(opts.cudaRoot, 'lib64') ;
  end

  % Set the nvcc method as default for Win platforms
  if strcmp(arch, 'win64') && isempty(opts.cudaMethod)
    opts.cudaMethod = 'nvcc';
  end

  % Activate the CUDA Devkit
  cuver = activate_nvcc(opts.nvccPath) ;
  opts.verbose && fprintf('%s:\tCUDA: using NVCC ''%s'' (%d).\n', ...
                          mfilename, opts.nvccPath, cuver) ;

  % Set the CUDA arch string (select GPU architecture)
  if isempty(opts.cudaArch), opts.cudaArch = get_cuda_arch(opts) ; end
  opts.verbose && fprintf('%s:\tCUDA: NVCC architecture string: ''%s''.\n', ...
                          mfilename, opts.cudaArch) ;
end

if opts.enableCudnn
  opts.cudnnIncludeDir = fullfile(opts.cudnnRoot, 'include') ;
  switch arch
    case 'win64', opts.cudnnLibDir = fullfile(opts.cudnnRoot, 'lib', 'x64') ;
    case 'maci64', opts.cudnnLibDir = fullfile(opts.cudnnRoot, 'lib') ;
    case 'glnxa64', opts.cudnnLibDir = fullfile(opts.cudnnRoot, 'lib64') ;
  end
end

% --------------------------------------------------------------------
%                                                     Compiler options
% --------------------------------------------------------------------

% Build directories
flags.src_dir = fullfile(root, 'matlab', 'src') ;
flags.mex_dir = fullfile(root, 'matlab', 'mex') ;
flags.bld_dir = fullfile(flags.mex_dir, '.build');
if ~exist(fullfile(flags.bld_dir,'bits','impl'), 'dir')
  mkdir(fullfile(flags.bld_dir,'bits','impl')) ;
end

% BASE: Base flags passed to `mex` and `nvcc` always.
flags.base = {} ;
if opts.enableGpu, flags.base{end+1} = '-DENABLE_GPU' ; end
if opts.enableDouble, flags.base{end+1} = '-DENABLE_DOUBLE' ; end
if opts.enableCudnn
  flags.base{end+1} = '-DENABLE_CUDNN' ;
  flags.base{end+1} = ['-I"' opts.cudnnIncludeDir '"'] ;
end
if opts.verbose > 1, flags.base{end+1} = '-v' ; end
if opts.debug
  flags.base{end+1} = '-g' ;
  flags.base{end+1} = '-DDEBUG' ;
else
  flags.base{end+1} = '-O' ;
  flags.base{end+1} = '-DNDEBUG' ;
end

% MEX: Additional flags passed to `mex` for compiling C++
% code. CXX and CXXOPTIOM are passed directly to the encapsualted compiler.
flags.mex = {'-largeArrayDims'} ;
flags.cxx = {} ;
flags.cxxoptim = {} ;
if ~isempty(opts.mexConfig), flags.mex = horzcat(flags.mex, {'-f', opts.mexConfig}) ; end

% MEX: Additional flags passed to `mex` for compiling CUDA
% code. CXX and CXXOPTIOM are passed directly to the encapsualted compiler.
flags.mexcuda = {'-largeArrayDims'} ;
flags.mexcuda_cxx = {} ;
flags.mexcuda_cxxoptim = {} ;
if ~isempty(opts.mexCudaConfig), flags.mexcuda = horzcat(flags.mexcuda, {'-f', opts.mexCudaConfig}) ; end

% MEX_LINK: Additional flags passed to `mex` for linking.
flags.mexlink = {'-largeArrayDims','-lmwblas'} ;
flags.mexlink_ldflags = {} ;
flags.mexlink_ldoptimflags = {} ;
flags.mexlink_linklibs = {} ;

% NVCC: Additional flags passed to `nvcc` for compiling CUDA code.
flags.nvcc = {'-D_FORCE_INLINES', '--std=c++11', ...
  sprintf('-I"%s"',fullfile(matlabroot,'extern','include')), ...
  sprintf('-I"%s"',fullfile(toolboxdir('distcomp'),'gpu','extern','include')), ...
  opts.cudaArch} ;

switch arch
  case {'maci64','glnxa64'}
    flags.cxx{end+1} = '--std=c++11' ;
    flags.nvcc{end+1} = '--compiler-options=-fPIC' ;
    if ~opts.debug
      flags.cxxoptim = horzcat(flags.cxxoptim,'-mssse3','-ffast-math') ;
      flags.mexcuda_cxxoptim{end+1} = '--compiler-options=-mssse3,-ffast-math' ;
      flags.nvcc{end+1} = '--compiler-options=-mssse3,-ffast-math' ;
    end
  case 'win64'
    % Visual Studio 2015 does C++11 without futher switches
end

if opts.enableGpu
  flags.mexlink = horzcat(flags.mexlink, ...
    {['-L"' opts.cudaLibDir '"'], '-lcudart', '-lcublas'}) ;
  switch arch
    case {'maci64', 'glnxa64'}
      flags.mexlink{end+1} = '-lmwgpu' ;
    case 'win64'
      flags.mexlink{end+1} = '-lgpu' ;
  end
  if opts.enableCudnn
    flags.mexlink{end+1} = ['-L"' opts.cudnnLibDir '"'] ;
    flags.mexlink{end+1} = '-lcudnn' ;
  end
end

switch arch
  case {'maci64'}
    flags.mex{end+1} = '-cxx' ;
    flags.nvcc{end+1} = '--compiler-options=-mmacosx-version-min=10.10' ;
    [s,r] = system('xcrun -f clang++') ;
    if s == 0
      flags.nvcc{end+1} = sprintf('--compiler-bindir="%s"',strtrim(r)) ;
    end
    if opts.enableGpu
      flags.mexlink_ldflags{end+1} = sprintf('-Wl,-rpath -Wl,"%s"', opts.cudaLibDir) ;
    end
    if opts.enableGpu && opts.enableCudnn
      flags.mexlink_ldflags{end+1} = sprintf('-Wl,-rpath -Wl,"%s"', opts.cudnnLibDir) ;
    end

  case {'glnxa64'}
    flags.mex{end+1} = '-cxx' ;
    flags.mexlink{end+1} = '-lrt' ;
    if opts.enableGpu
      flags.mexlink_ldflags{end+1} = sprintf('-Wl,-rpath -Wl,"%s"', opts.cudaLibDir) ;
    end
    if opts.enableGpu && opts.enableCudnn
      flags.mexlink_ldflags{end+1} = sprintf('-Wl,-rpath -Wl,"%s"', opts.cudnnLibDir) ;
    end

  case {'win64'}
    % VisualC does not pass this even if available in the CPU architecture
    flags.mex{end+1} = '-D__SSSE3__' ;
    cl_path = fileparts(check_clpath()); % check whether cl.exe in path
    flags.nvcc{end+1} = '--compiler-options=/MD' ;
    flags.nvcc{end+1} = sprintf('--compiler-bindir="%s"', cl_path) ;
end

if opts.enableImreadJpeg
  flags.mex = horzcat(flags.mex, opts.imageLibraryCompileFlags) ;
  flags.mexlink_linklibs = horzcat(flags.mexlink_linklibs, opts.imageLibraryLinkFlags) ;
end

% --------------------------------------------------------------------
%                                                          Command flags
% --------------------------------------------------------------------

if opts.verbose
  fprintf('%s: * Compiler and linker configurations *\n', mfilename) ;
  fprintf('%s: \tintermediate build products directory: %s\n', mfilename, flags.bld_dir) ;
  fprintf('%s: \tMEX files: %s/\n', mfilename, flags.mex_dir) ;
  fprintf('%s: \tBase options: %s\n', mfilename, strjoin(flags.base)) ;
  fprintf('%s: \tMEX CXX: %s\n', mfilename, strjoin(flags.mex)) ;
  fprintf('%s: \tMEX CXXFLAGS: %s\n', mfilename, strjoin(flags.cxx)) ;
  fprintf('%s: \tMEX CXXOPTIMFLAGS: %s\n', mfilename, strjoin(flags.cxxoptim)) ;
  fprintf('%s: \tMEX LINK: %s\n', mfilename, strjoin(flags.mexlink)) ;
  fprintf('%s: \tMEX LINK LDFLAGS: %s\n', mfilename, strjoin(flags.mexlink_ldflags)) ;
  fprintf('%s: \tMEX LINK LDOPTIMFLAGS: %s\n', mfilename, strjoin(flags.mexlink_ldoptimflags)) ;
  fprintf('%s: \tMEX LINK LINKLIBS: %s\n', mfilename, strjoin(flags.mexlink_linklibs)) ;
end
if opts.verbose && opts.enableGpu
  fprintf('%s: \tMEX CUDA: %s\n', mfilename, strjoin(flags.mexcuda)) ;
  fprintf('%s: \tMEX CUDA CXXFLAGS: %s\n', mfilename, strjoin(flags.mexcuda_cxx)) ;
  fprintf('%s: \tMEX CUDA CXXOPTIMFLAGS: %s\n', mfilename, strjoin(flags.mexcuda_cxxoptim)) ;
end
if opts.verbose && opts.enableGpu && strcmp(opts.cudaMethod,'nvcc')
  fprintf('%s: \tNVCC: %s\n', mfilename, strjoin(flags.nvcc)) ;
end
if opts.verbose && opts.enableImreadJpeg
  fprintf('%s: * Reading images *\n', mfilename) ;
  fprintf('%s: \tvl_imreadjpeg enabled\n', mfilename) ;
  fprintf('%s: \timage library: %s\n', mfilename, opts.imageLibrary) ;
  fprintf('%s: \timage library compile flags: %s\n', mfilename, strjoin(opts.imageLibraryCompileFlags)) ;
  fprintf('%s: \timage library link flags: %s\n', mfilename, strjoin(opts.imageLibraryLinkFlags)) ;
end

% --------------------------------------------------------------------
%                                                              Compile
% --------------------------------------------------------------------

% Apply pre-compilation modifier function to adjust the flags and
% parameters. This can be used to add additional files to compile on the
% fly.
if ~isempty(opts.preCompileFn)
  [opts, mex_src, lib_src, flags] = opts.preCompileFn(opts, mex_src, lib_src, flags) ;
end

% Compile intermediate object files
srcs = horzcat(lib_src,mex_src) ;
for i = 1:numel(horzcat(lib_src, mex_src))
  [~,~,ext] = fileparts(srcs{i}) ; ext(1) = [] ;
  objfile = toobj(flags.bld_dir,srcs{i});
  if strcmp(ext,'cu')
    if strcmp(opts.cudaMethod,'nvcc')
      nvcc_compile(opts, srcs{i}, objfile, flags) ;
    else
      mexcuda_compile(opts, srcs{i}, objfile, flags) ;
    end
  else
    mex_compile(opts, srcs{i}, objfile, flags) ;
  end
  assert(exist(objfile, 'file') ~= 0, 'Compilation of %s failed.', objfile);
end

% Link MEX files
for i = 1:numel(mex_src)
  objs = toobj(flags.bld_dir, [mex_src(i), lib_src]) ;
  mex_link(opts, objs, flags.mex_dir, flags) ;
end

% Reset path adding the mex subdirectory just created
vl_setupnn() ;

if strcmp(arch, 'win64') && opts.enableCudnn
  if opts.verbose(), fprintf('Copying CuDNN dll to mex folder.\n'); end
  copyfile(fullfile(opts.cudnnRoot, 'bin', '*.dll'), flags.mex_dir);
end

% Save the last compile flags to the build dir
if isempty(opts.preCompileFn)
  save(fullfile(flags.bld_dir, 'last_compile_opts.mat'), '-struct', 'opts');
end

% --------------------------------------------------------------------
%                                                    Utility functions
% --------------------------------------------------------------------

% --------------------------------------------------------------------
function check_compability(arch)
% --------------------------------------------------------------------
cc = mex.getCompilerConfigurations('C++');
if isempty(cc)
  error(['Mex is not configured.'...
    'Run "mex -setup" to configure your compiler. See ',...
    'http://www.mathworks.com/support/compilers ', ...
    'for supported compilers for your platform.']);
end

switch arch
  case 'win64'
    clversion = str2double(cc.Version);
    if clversion < 14
      error('Unsupported VS C++ compiler, ver >=14.0 required (VS 2015).');
    end
  case 'maci64'
  case 'glnxa64'
  otherwise, error('Unsupported architecture ''%s''.', arch) ;
end

% --------------------------------------------------------------------
function done = check_deps(opts, tgt, src)
% --------------------------------------------------------------------
done = false ;
if ~iscell(src), src = {src} ; end
if ~opts.continue, return ; end
if ~exist(tgt,'file'), return ; end
ttime = dir(tgt) ; ttime = ttime.datenum ;
for i=1:numel(src)
  stime = dir(src{i}) ; stime = stime.datenum ;
  if stime > ttime, return ; end
end
fprintf('%s: ''%s'' already there, skipping.\n', mfilename, tgt) ;
done = true ;

% --------------------------------------------------------------------
function objs = toobj(bld_dir, srcs)
% --------------------------------------------------------------------
str = [filesep, 'src', filesep]; % NASTY. Do with regexp?
multiple = iscell(srcs) ;
if ~multiple, srcs = {srcs} ; end
objs = cell(1, numel(srcs));
for t = 1:numel(srcs)
  i = strfind(srcs{t},str);
  i = i(end); % last occurence of '/src/'
  objs{t} = fullfile(bld_dir, srcs{t}(i+numel(str):end)) ;
end
if ~multiple, objs = objs{1} ; end
objs = regexprep(objs,'.cpp$',['.' objext]) ;
objs = regexprep(objs,'.cu$',['.' objext]) ;
objs = regexprep(objs,'.c$',['.' objext]) ;

% --------------------------------------------------------------------
function mex_compile(opts, src, tgt, flags)
% --------------------------------------------------------------------
if check_deps(opts, tgt, src), return ; end
args = horzcat({'-c', '-outdir', fileparts(tgt), src}, ...
  flags.base, flags.mex, ...
  {['CXXFLAGS=$CXXFLAGS ' strjoin(flags.cxx)]}, ...
  {['CXXOPTIMFLAGS=$CXXOPTIMFLAGS ' strjoin(flags.cxxoptim)]}) ;
opts.verbose && fprintf('%s: MEX CC: %s\n', mfilename, strjoin(args)) ;
mex(args{:}) ;

% --------------------------------------------------------------------
function mexcuda_compile(opts, src, tgt, flags)
% --------------------------------------------------------------------
if check_deps(opts, tgt, src), return ; end
% Hacky fix: In glnxa64 MATLAB includes the -ansi option by default, which
% prevents -std=c++11 to work (an error?). This could be solved by editing the
% mex configuration file; for convenience, we take it out here by
% avoiding to append to the default flags.
glue = '$CXXFLAGS' ;
switch computer('arch')
  case {'glnxa64'}
    glue = '--compiler-options=-fexceptions,-fPIC,-fno-omit-frame-pointer,-pthread' ;
end
args = horzcat({'-c', '-outdir', fileparts(tgt), src}, ...
  flags.base, flags.mexcuda, ...
  {['CXXFLAGS=' glue ' ' strjoin(flags.mexcuda_cxx)]}, ...
  {['CXXOPTIMFLAGS=$CXXOPTIMFLAGS ' strjoin(flags.mexcuda_cxxoptim)]}) ;
opts.verbose && fprintf('%s: MEX CUDA: %s\n', mfilename, strjoin(args)) ;
mexcuda(args{:}) ;

% --------------------------------------------------------------------
function nvcc_compile(opts, src, tgt, flags)
% --------------------------------------------------------------------
if check_deps(opts, tgt, src), return ; end
nvcc_path = fullfile(opts.cudaRoot, 'bin', 'nvcc');
nvcc_cmd = sprintf('"%s" -c -o "%s" "%s" %s ', ...
                   nvcc_path, tgt, src, ...
                   strjoin(horzcat(flags.base,flags.nvcc)));
opts.verbose && fprintf('%s: NVCC CC: %s\n', mfilename, nvcc_cmd) ;
status = system(nvcc_cmd);
if status, error('Command %s failed.', nvcc_cmd); end;

% --------------------------------------------------------------------
function mex_link(opts, objs, mex_dir, flags)
% --------------------------------------------------------------------
args = horzcat({'-outdir', mex_dir}, ...
  flags.base, flags.mexlink, ...
  {['LDFLAGS=$LDFLAGS ' strjoin(flags.mexlink_ldflags)]}, ...
  {['LDOPTIMFLAGS=$LDOPTIMFLAGS ' strjoin(flags.mexlink_ldoptimflags)]}, ...
  {['LINKLIBS=' strjoin(flags.mexlink_linklibs) ' $LINKLIBS']}, ...
  objs) ;
opts.verbose && fprintf('%s: MEX LINK: %s\n', mfilename, strjoin(args)) ;
mex(args{:}) ;

% --------------------------------------------------------------------
function ext = objext()
% --------------------------------------------------------------------
% Get the extension for an 'object' file for the current computer
% architecture
switch computer('arch')
  case 'win64', ext = 'obj';
  case {'maci64', 'glnxa64'}, ext = 'o' ;
  otherwise, error('Unsupported architecture %s.', computer) ;
end

% --------------------------------------------------------------------
function cl_path = check_clpath()
% --------------------------------------------------------------------
% Checks whether the cl.exe is in the path (needed for the nvcc). If
% not, tries to guess the location out of mex configuration.
cc = mex.getCompilerConfigurations('c++');
cl_path = fullfile(cc.Location, 'VC', 'bin', 'amd64');
[status, ~] = system('cl.exe -help');
if status == 1
  % Add cl.exe to system path so that nvcc can find it.
  warning('CL.EXE not found in PATH. Trying to guess out of mex setup.');
  prev_path = getenv('PATH');
  setenv('PATH', [prev_path ';' cl_path]);
  status = system('cl.exe');
  if status == 1
    setenv('PATH', prev_path);
    error('Unable to find cl.exe');
  else
    fprintf('Location of cl.exe (%s) successfully added to your PATH.\n', ...
      cl_path);
  end
end

% -------------------------------------------------------------------------
function paths = which_nvcc()
% -------------------------------------------------------------------------
switch computer('arch')
  case 'win64'
    [~, paths] = system('where nvcc.exe');
    paths = strtrim(paths);
    paths = paths(strfind(paths, '.exe'));
  case {'maci64', 'glnxa64'}
    [~, paths] = system('which nvcc');
    paths = strtrim(paths) ;
end

% -------------------------------------------------------------------------
function cuda_root = search_cuda_devkit(opts)
% -------------------------------------------------------------------------
% This function tries to to locate a working copy of the CUDA Devkit.

opts.verbose && fprintf(['%s:\tCUDA: searching for the CUDA Devkit' ...
                    ' (use the option ''CudaRoot'' to override):\n'], mfilename);

% Propose a number of candidate paths for NVCC
paths = {getenv('MW_NVCC_PATH')} ;
paths = [paths, which_nvcc()] ;
for v = {'5.5', '6.0', '6.5', '7.0', '7.5', '8.0', '8.5', '9.0', '9.5', '10.0'}
  switch computer('arch')
    case 'glnxa64'
      paths{end+1} = sprintf('/usr/local/cuda-%s/bin/nvcc', char(v)) ;
    case 'maci64'
      paths{end+1} = sprintf('/Developer/NVIDIA/CUDA-%s/bin/nvcc', char(v)) ;
    case 'win64'
      paths{end+1} = sprintf('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v%s\\bin\\nvcc.exe', char(v)) ;
  end
end
paths{end+1} = sprintf('/usr/local/cuda/bin/nvcc') ;

% Validate each candidate NVCC path
for i=1:numel(paths)
  nvcc(i).path = paths{i} ;
  [nvcc(i).isvalid, nvcc(i).version] = validate_nvcc(paths{i}) ;
end
if opts.verbose
  fprintf('\t| %5s | %5s | %-70s |\n', 'valid', 'ver', 'NVCC path') ;
  for i=1:numel(paths)
    fprintf('\t| %5d | %5d | %-70s |\n', ...
            nvcc(i).isvalid, nvcc(i).version, nvcc(i).path) ;
  end
end

% Pick an entry
index = find([nvcc.isvalid]) ;
if isempty(index)
  error('Could not find a valid NVCC executable\n') ;
end
[~, newest] = max([nvcc(index).version]);
nvcc = nvcc(index(newest)) ;
cuda_root = fileparts(fileparts(nvcc.path)) ;

if opts.verbose
  fprintf('%s:\tCUDA: choosing NVCC compiler ''%s'' (version %d)\n', ...
          mfilename, nvcc.path, nvcc.version) ;
end

% -------------------------------------------------------------------------
function [valid, cuver]  = validate_nvcc(nvccPath)
% -------------------------------------------------------------------------
[status, output] = system(sprintf('"%s" --version', nvccPath)) ;
valid = (status == 0) ;
if ~valid
  cuver = 0 ;
  return ;
end
match = regexp(output, 'V(\d+\.\d+\.\d+)', 'match') ;
if isempty(match), valid = false ; return ; end
cuver = [1e4 1e2 1] * sscanf(match{1}, 'V%d.%d.%d') ;

% --------------------------------------------------------------------
function cuver = activate_nvcc(nvccPath)
% --------------------------------------------------------------------

% Validate the NVCC compiler installation
[valid, cuver] = validate_nvcc(nvccPath) ;
if ~valid
  error('The NVCC compiler ''%s'' does not appear to be valid.', nvccPath) ;
end

% Make sure that NVCC is visible by MEX by setting the MW_NVCC_PATH
% environment variable to the NVCC compiler path
if ~strcmp(getenv('MW_NVCC_PATH'), nvccPath)
  warning('Setting the ''MW_NVCC_PATH'' environment variable to ''%s''', nvccPath) ;
  setenv('MW_NVCC_PATH', nvccPath) ;
end

% In some operating systems and MATLAB versions, NVCC must also be
% available in the command line search path. Make sure that this is%
% the case.
[valid_, cuver_] = validate_nvcc('nvcc') ;
if ~valid_ || cuver_ ~= cuver
  warning('NVCC not found in the command line path or the one found does not matches ''%s''.', nvccPath);
  nvccDir = fileparts(nvccPath) ;
  prevPath = getenv('PATH') ;
  switch computer
    case 'PCWIN64', separator = ';' ;
    case {'GLNXA64', 'MACI64'}, separator = ':' ;
  end
  setenv('PATH', [nvccDir separator prevPath]) ;
  [valid_, cuver_] = validate_nvcc('nvcc') ;
  if ~valid_ || cuver_ ~= cuver
    setenv('PATH', prevPath) ;
    error('Unable to set the command line path to point to ''%s'' correctly.', nvccPath) ;
  else
    fprintf('Location of NVCC (%s) added to your command search PATH.\n', nvccDir) ;
  end
end

% --------------------------------------------------------------------
function cudaArch = get_cuda_arch(opts)
% --------------------------------------------------------------------
opts.verbose && fprintf('%s:\tCUDA: determining GPU compute capability (use the ''CudaArch'' option to override)\n', mfilename);
try
  gpu_device = gpuDevice();
  arch = str2double(strrep(gpu_device.ComputeCapability, '.', ''));
  supparchs = get_nvcc_supported_archs(opts.nvccPath);
  [~, archi] = max(min(supparchs - arch, 0));
  arch_code = num2str(supparchs(archi));
  assert(~isempty(arch_code));
  cudaArch = ...
      sprintf('-gencode=arch=compute_%s,code=\\\"sm_%s,compute_%s\\\" ', ...
              arch_code, arch_code, arch_code) ;
catch
  opts.verbose && fprintf(['%s:\tCUDA: cannot determine the capabilities of the installed GPU and/or CUDA; ' ...
                      'falling back to default\n'], mfilename);
  cudaArch = opts.defCudaArch;
end

% --------------------------------------------------------------------
function archs = get_nvcc_supported_archs(nvccPath)
% --------------------------------------------------------------------
switch computer('arch')
  case {'win64'}
    [status, hstring] = system(sprintf('"%s" --help',nvccPath));
  otherwise
    % fix possible output corruption (see manual)
    [status, hstring] = system(sprintf('"%s" --help < /dev/null',nvccPath)) ;
end
archs = regexp(hstring, '''sm_(\d{2})''', 'tokens');
archs = cellfun(@(a) str2double(a{1}), archs);
if status, error('NVCC command failed: %s', hstring); end;
