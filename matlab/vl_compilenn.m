function vl_compilenn( varargin )
% VL_COMPILENN  Compile the MatConvNet toolbox
%    The `vl_compilenn()` function compiles the MEX files in the
%    MatConvNet toolbox. See below for the requirements for compiling
%    CPU and GPU code, respectively.
%
%    `vl_compilenn('OPTION', ARG, ...)` accepts the following options:
%
%    `EnableGpu`:: `false`
%       Set to true in order to enable GPU support.
%
%    `Verbose`:: 0
%       Set the verbosity level, 0, 1 or 2.
%
%    `Debug`:: `false`
%       Set to true to compile the binaries with debugging
%       information.
%
%    `CudaMethod`:: Linux & Mac OSX: `mex`; Windows: `nvcc`
%       Choose the method used to compile CUDA code. There are two
%       methods:
%
%       * The **`mex`** method uses the MATLAB MEX command with the
%         configuration file
%         `<matconvnet_root>/matlab/src/config/mex_CUDA_<arch>.[sh/xml]`
%         This configuration file is in XML format since MATLAB 8.3
%         (R2014a) and is a Shell script for earlier version. This is,
%         principle, the preferred method as it uses MATLAB only.
%
%       * The **`nvcc`** method calls the NVIDIA CUDA compiler `nvcc`
%         directly to compile CUDA source code into object files.
%
%         In some cases this method allows to use a CUDA Devkit
%         version that is not the one that officially supported by a
%         particular MATALB version (see below). It is also the
%         default method for compilation under Windows.
%
%         With the `nvcc` method, you can specify different CUDA root
%         directory (e.g. for compiling with different CUDA version)
%         with the `CudaRoot` option.
%
%    `CudaRoot`:: guessed automatically
%       This option specifies the path to the CUDA Devkit to use
%       for compilation.
%
%    `EnableImreadJpeg`:: `false`
%       Set true to compile `vl_imreadjpeg()`. In order to successfully
%       compile, libjpeg must be in linker search path, or the option
%       `ImreadJpegFlags` must be adjusted appropriately.
%
%    `ImreadJpegFlags`:: `{'-ljpeg'}`
%       Specify additional flags to compile `vl_imreadjpeg`. This
%       function currently requires libjpeg.
%
%    ## Compiling the CPU code
%
%    By default, the `EnableGpu` option is switched to off, such that
%    the GPU code support is not compiled in.
%
%    Generally, you only need a C/C++ compiler (usually Xcode, GCC or
%    Visual Studio for Mac, Linux, and Windows respectively). The
%    compiler can be setup in MATLAB using the
%
%       mex -setup
%
%    command.
%
%    ## Compiling the GPU code
%
%    In order to compile the GPU code, set the `EnableGpu` option to
%    `true`. For this to work you will need:
%
%    * To satisfy all the requirement to compile the CPU code (see
%      above).
%
%    * A NVIDIA GPU with at least *compute capability 2.0*.
%
%    * The *MATALB Parallel Computing Toolbox*. This can be purchased
%      from Mathworks (type `ver` in MATLAB to see if this toolbox is
%      already comprised in your MATLAB installation; it often is).
%
%    * A copy of the *CUDA Devkit*, which can be downloaded for free
%      from NVIDIA. Note that each MATLAB version requires a
%      particular CUDA Devkit version:
%
%      | MATLAB version | Release | CUDA Devkit |
%      |----------------|---------|-------------|
%      | 2013b          | 2013b   | 5.5         |
%      | 2014a          | 2014a   | 5.5         |
%      | 2014b          | 2014b   | 6.0         |
%
%      A different versions of CUDA may work using the hack described
%      above (i.e. setting the `CudaMethod` to `nvcc`).
%
%    The following configurations have been tested successfully:
%
%    * Windows 7 x64, MATLAB R2014a, Visual C++ 2010 and CUDA Toolkit
%      6.5 (unable to compile with Visual C++ 2013).
%    * Windows 8 x64, MATLAB R2014a, Visual C++ 2013 and CUDA
%      Toolkit 6.5.
%    * Mac OS X 10.9 and 10.10, MATLAB R2013a and R2013b, Xcode, CUDA
%      Toolkit 5.5.
%    * GNU/Linux, MATALB R2014a, gcc, CUDA Toolkit 5.5.
%
%    Furthermore your GPU card must have ComputeCapability >= 2.0 (see
%    output of `gpudevice()`) in order to be able to run the GPU code.
%    To change the compute capabilities, for `mex` `CudaMethod` edit
%    the particular config file.  For the 'nvcc' method, compute
%    capability is guessed based on the GPUDEVICE output. You can
%    override it by setting the 'CudaArch' parameter (e.g. in case of
%    multiple GPUs with various architectures).
%
%    See also: `vl_setup()`, `vl_imreadjpeg()`, [Compiling MEX files
%    containing CUDA
%    code](http://mathworks.com/help/distcomp/run-mex-functions-containing-cuda-code.html)

% Copyright (C) 2014 Karel Lenc and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

% Get MatConvNet root directory
root = fileparts(fileparts(mfilename('fullpath'))) ;
run(fullfile(root, 'matlab', 'vl_setupnn.m')) ;

% --------------------------------------------------------------------
%                                                        Parse options
% --------------------------------------------------------------------

opts.enableGpu        = false;
opts.enableImreadJpeg = false;
opts.imreadJpegFlags  = {'-ljpeg'};
opts.verbose          = 0;
opts.debug            = false;
opts.cudaMethod       = [] ;
opts.cudaRoot         = [] ;
opts.cudaArch         = [] ;
opts.defCudaArch      = [...
  '-gencode=arch=compute_20,code=\"sm_20,compute_20\" '...
  '-gencode=arch=compute_30,code=\"sm_30,compute_30\"'];
opts = vl_argparse(opts, varargin);

% --------------------------------------------------------------------
%                                                     Files to compile
% --------------------------------------------------------------------

cpp_src={...
  fullfile(root, 'matlab', 'src', 'bits', 'im2col.cpp'), ...
  fullfile(root, 'matlab', 'src', 'bits', 'pooling.cpp'), ...
  fullfile(root, 'matlab', 'src', 'bits', 'normalize.cpp'), ...
  fullfile(root, 'matlab', 'src', 'bits', 'subsample.cpp')} ;
mex_src={...
  fullfile(root, 'matlab', 'src', 'vl_nnconv.cpp'), ...
  fullfile(root, 'matlab', 'src', 'vl_nnpool.cpp'), ...
  fullfile(root, 'matlab', 'src', 'vl_nnnormalize.cpp')} ;
cu_src={...
  fullfile(root, 'matlab', 'src', 'bits', 'im2col_gpu.cu'), ...
  fullfile(root, 'matlab', 'src', 'bits', 'pooling_gpu.cu'), ...
  fullfile(root, 'matlab', 'src', 'bits', 'normalize_gpu.cu'), ...
  fullfile(root, 'matlab', 'src', 'bits', 'subsample_gpu.cu')} ;
mex_cu_src={...
  fullfile(root, 'matlab', 'src', 'vl_nnconv.cu'), ...
  fullfile(root, 'matlab', 'src', 'vl_nnpool.cu'), ...
  fullfile(root, 'matlab', 'src', 'vl_nnnormalize.cu')} ;

% --------------------------------------------------------------------
%                                                     Compiler options
% --------------------------------------------------------------------

% Build directories
mex_dir = fullfile(root, 'matlab', 'mex') ;
bld_dir = fullfile(mex_dir, '.build');
if ~exist(bld_dir, 'dir'), mkdir(bld_dir); end

% MEX arguments
arch = computer('arch') ;
mex_libs = {'-lmwblas'};
mex_opts = {'-largeArrayDims'};
if opts.verbose > 1, mex_opts{end+1} = '-v'; end
if opts.debug, mex_opts{end+1} = '-g' ; end

if opts.verbose
  fprintf('%s: intermediate build products: %s\n', mfilename, bld_dir) ;
  fprintf('%s: MEX files: %s/\n', mfilename, mex_dir) ;
  fprintf('%s: MEX compiler options: %s\n', mfilename, strjoin(mex_opts)) ;
  fprintf('%s: MEX linker options: %s\n', mfilename, strjoin(mex_libs)) ;
end

% CUDA MEX and NVCC arguments
opts.verbose && fprintf('%s: enable GPU: %d\n', mfilename, opts.enableGpu) ;
if opts.enableGpu

  % Get CUDA Devkit and other CUDA paths
  if isempty(opts.cudaRoot), opts.cudaRoot = search_cuda_devkit(opts); end
  check_nvcc(opts.cudaRoot);
  opts.verbose && fprintf('%s:\tCUDA: using CUDA Devkit ''%s''.\n', ...
                          mfilename, opts.cudaRoot) ;
  switch arch
    case 'win64', opts.cudaLibDir = fullfile(opts.cudaRoot, 'lib', 'x64') ;
    case 'maci64', opts.cudaLibDir = fullfile(opts.cudaRoot, 'lib') ;
    case 'glnxa64', opts.cudaLibDir = fullfile(opts.cudaRoot, 'lib64') ;
    otherwise, error('Unsupported architecture ''%s''.', arch) ;
  end
  opts.nvccPath = fullfile(opts.cudaRoot, 'bin', 'nvcc') ;

  % Get CUDA arch string (GPU capabilities)
  if isempty(opts.cudaArch), opts.cudaArch = get_cuda_arch(opts) ; end
  opts.verbose && fprintf('%s:\tCUDA: CUDA NVCC arch string: ''%s''.\n', ...
                          mfilename, opts.cudaArch) ;

  % Get CUDA compilation method
  if isempty(opts.cudaMethod)
    switch arch
      case 'win64', opts.cudaMethod = 'nvcc' ;
      case {'maci64', 'glnxa64'}, opts.cudaMethod = 'mex' ;
    end
  end

  % Get CUDA flags
  mex_cu_libs = {mex_libs{:}, ...
                 ['-L' fullfile(matlabroot, 'bin', arch)], ...
                 ['-L' opts.cudaLibDir], ...
                 '-lcudart', ...
                 '-lcublas'};
  if strcmp(computer, 'PCWIN64')
    mex_cu_libs{end+1} = '-lgpu';
  end
  mex_cu_opts = mex_opts ;
  mex_cu_opts{end+1} = '-DENABLE_GPU' ;
  switch opts.cudaMethod
    case 'mex'
      mex_cu_opts = [mex_cu_opts {'-f' mex_cuda_config(root)}];
      mex_cu_opts{end+1} = ['NVCCFLAGS=' opts.cudaArch '$NVCC_FLAGS'] ;
      if ~strcmp(getenv('MW_NVCC_PATH'), opts.nvccPath)
        warning('Setting the ''MW_NVCC_PATH'' environment variable to ''%s''', ...
                opts.nvccPath) ;
        setenv('MW_NVCC_PATH', opts.nvccPath) ;
      end

    case 'nvcc'
      mex_cu_opts = [mex_cu_opts {'-cxx'}];
      switch arch
        case 'maci64'
          mex_cu_opts{end+1} = 'LDFLAGS=$LDFLAGS -stdlib=libstdc++' ;
          mex_cu_opts{end+1} = 'CXXFLAGS=$CXXFLAGS -stdlib=libstdc++' ;
        case 'glnxa64'
          mex_cu_libs{end+1} = '-lmwgpu' ;
      end
  end

  if opts.verbose
    fprintf('%s:\tCUDA: compilation method: %s\n', mfilename, opts.cudaMethod) ;
    fprintf('%s:\tCUDA: MEX compiler options: %s\n', mfilename, strjoin(mex_cu_opts)) ;
    fprintf('%s:\tCUDA: MEX linker options: %s\n', mfilename, strjoin(mex_cu_libs)) ;
  end

  if strcmp(opts.cudaMethod, 'nvcc')
    nvcc_opts = nvcc_get_opts(opts) ;
    opts.verbose && fprintf('%s:\tCUDA: NVCC compiler options: %s\n', ...
                            mfilename, nvcc_opts) ;
  end
end

% --------------------------------------------------------------------
%                                                              Compile
% --------------------------------------------------------------------

% Compile CPP files
obj_files = mex_compile(opts, cpp_src, bld_dir, mex_opts);

if ~opts.enableGpu
  % Compile and link CPP MEX files
  mex_link(opts, mex_src, obj_files, mex_libs, mex_dir, mex_opts);
else
  % Compile CUDA MEX files
  switch opts.cudaMethod
    case 'mex'
      cuobj_files = mex_compile(opts, cu_src, bld_dir, mex_cu_opts);
      mss = mex_cu_src;
    case 'nvcc'
      cuobj_files = nvcc_compile(opts, cu_src, bld_dir, nvcc_opts);
      mss = nvcc_compile(opts, mex_cu_src, bld_dir, nvcc_opts);
  end

  % Link/compile CUDA MEX files
  mex_link(opts, ...
           mss, ...
           [obj_files cuobj_files], ...
           mex_cu_libs, ...
           mex_dir, mex_cu_opts);
end

if opts.enableImreadJpeg
  imr_src = fullfile(root, 'matlab', 'src', 'vl_imreadjpeg.c');
  mex_link(opts, {imr_src}, {}, [mex_libs opts.imreadJpegFlags], mex_dir, mex_opts);
end

% Reset path adding the mex subdirectory just created
vl_setupnn() ;

% --------------------------------------------------------------------
%                                                           MEX recipe
% --------------------------------------------------------------------

% --------------------------------------------------------------------
function objs = mex_compile(opts, srcs, bld_dir, mex_opts)
% --------------------------------------------------------------------
for i=1:numel(srcs)
  mopts = {'-outdir', bld_dir, srcs{i}, '-c', mex_opts{:}} ;
  opts.verbose && fprintf('%s: compiling: mex %s\n', mfilename, strjoin(mopts)) ;
  mex(mopts{:}) ;
  [~,base] = fileparts(srcs{i}) ;
  objs{i} = fullfile(bld_dir, [base '.' objext]) ;
end

% --------------------------------------------------------------------
function mex_link(opts, srcs, objs, libs, mex_dir, mex_opts)
% --------------------------------------------------------------------
for i=1:numel(srcs)
  mopts = {'-outdir', mex_dir, mex_opts{:}, srcs{i}, objs{:}, libs{:}} ;
  opts.verbose && fprintf('%s: linking: mex %s\n', mfilename, strjoin(mopts)) ;
  mex(mopts{:}) ;
end

% --------------------------------------------------------------------
%                                                          NVCC recipe
% --------------------------------------------------------------------

% --------------------------------------------------------------------
function nvcc_opts = nvcc_get_opts(opts)
% --------------------------------------------------------------------

nvcc_opts = [...
  opts.cudaArch, ' -DENABLE_GPU' ...
  ' -I"' fullfile(matlabroot, 'extern', 'include') '"' ...
  ' -I"', fullfile(matlabroot, 'toolbox','distcomp','gpu','extern','include') '"'];
if opts.verbose > 1
  nvcc_opts = [nvcc_opts ' -v'];
end
if opts.debug
  nvcc_opts = [nvcc_opts ' -g -DDEBUG'];
else
  nvcc_opts = [nvcc_opts ' -O3 -DNDEBUG'];
end

% System specific options
switch computer
  case 'PCWIN64'
    nvcc_opts = [nvcc_opts '  -Xcompiler  /MD']; % Use dynamic linker
    check_clpath(); % check whether cl.exe in path
  case {'MACI64', 'GLNXA64'}
    nvcc_opts = [nvcc_opts ' -Xcompiler -fPIC'];
end

% --------------------------------------------------------------------
function objs = nvcc_compile(opts, srcs, bld_dir, nvcc_opts)
% --------------------------------------------------------------------
nvcc_path = fullfile(opts.cudaRoot, 'bin', 'nvcc');
for i=1:numel(srcs)
  [~,base] = fileparts(srcs{i}) ;
  objs{i} = fullfile(bld_dir, [base '.' objext]) ;
  nvcc_cmd = sprintf('"%s" -c "%s" %s -o "%s"', ...
                     nvcc_path, srcs{i}, ...
                     nvcc_opts, objs{i});
  opts.verbose && fprintf('%s: CUDA: %s\n', mfilename, nvcc_cmd) ;
  status = system(nvcc_cmd);
  if status, error('Command %s failed.', nvcc_cmd); end;
end

% --------------------------------------------------------------------
%                                                    Utility functions
% --------------------------------------------------------------------

% --------------------------------------------------------------------
function ext = objext()
% --------------------------------------------------------------------
% Get the extension for an 'object' file for the current computer
% architecture
switch computer
  case 'PCWIN64', ext = 'obj';
  case {'MACI64', 'GLNXA64'}, ext = 'o' ;
  otherwise, error('Unsupported architecture %s.', computer) ;
end

% --------------------------------------------------------------------
function conf_file = mex_cuda_config(root)
% --------------------------------------------------------------------
% Get mex CUDA config file
mver = [1e4 1e2 1] * sscanf(version, '%d.%d.%d') ;
if mver <= 80200, ext = 'sh' ; else ext = 'xml' ; end
arch = computer('arch') ;
switch arch
  case {'win64'}
    config_dir = fullfile(matlabroot, 'toolbox', ...
                          'distcomp', 'gpu', 'extern', ...
                          'src', 'mex', arch) ;
  case {'maci64', 'glnxa64'}
    config_dir = fullfile(root, 'matlab', 'src', 'config') ;
end
conf_file = fullfile(config_dir, ['mex_CUDA_' arch '.' ext]);
fprintf('%s:\tCUDA: MEX config file: ''%s''\n', mfilename, conf_file);

% --------------------------------------------------------------------
function check_clpath()
% --------------------------------------------------------------------
% Checks whether the cl.exe is in the path (needed for the nvcc). If
% not, tries to guess the location out of mex configuration.
status = system('cl.exe -help');
if status == 1
  warning('CL.EXE not found in PATH. Trying to guess out of mex setup.');
  cc = mex.getCompilerConfigurations('c++');
  if isempty(cc)
    error('Mex is not configured. Run "mex -setup".');
  end
  prev_path = getenv('PATH');
  cl_path = fullfile(cc.Location, 'VC','bin','x86_amd64');
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
function paths = which_nvcc(opts)
% -------------------------------------------------------------------------
switch computer('arch')
  case 'win64'
    [~, paths] = system('where nvcc.exe');
    paths = strtrim(paths);
    paths = patsh(strfind(paths, '.exe'));
  case {'maci64', 'glnxa64'}
    [~, paths] = system('which nvcc');
    paths = strtrim(paths) ;
end

% -------------------------------------------------------------------------
function cuda_root = search_cuda_devkit(opts)
% -------------------------------------------------------------------------
% This function tries to to locate a working copy of the CUDA Devkit.

opts.verbose && fprintf(['%s:\tCUDA: seraching for the CUDA Devkit' ...
                    ' (use the option ''CudaRoot'' to override):\n'], mfilename);

% Propose a number of candidate paths for NVCC
paths = {getenv('MW_NVCC_PATH')} ;
paths = [paths, which_nvcc(opts)] ;
for v = {'5.5', '6.0', '6.5', '7.0'}
  paths{end+1} = sprintf('/Developer/NVIDIA/CUDA-%s/bin/nvcc', char(v)) ;
  paths{end+1} = sprintf('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v%s', char(v)) ;
end
paths{end+1} = sprintf('/usr/local/cuda/bin/nvcc') ;

% Validate each candidate NVCC path
for i=1:numel(paths)
  nvcc(i).path = paths{i} ;
  [nvcc(i).isvalid, nvcc(i).version] = validate_nvcc(opts,paths{i}) ;
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
nvcc = nvcc(index(1)) ;
cuda_root = fileparts(fileparts(nvcc.path)) ;

if opts.verbose
  fprintf('%s:\tCUDA: choosing NVCC compiler ''%s'' (version %d)\n', ...
          mfilename, nvcc.path, nvcc.version) ;
end

% -------------------------------------------------------------------------
function [valid, cuver]  = validate_nvcc(opts, nvcc_path)
% -------------------------------------------------------------------------
valid = false ;
cuver = 0 ;
if ~isempty(nvcc_path)
  [status, output] = system(sprintf('%s --version', nvcc_path)) ;
  valid = (status == 0) ;
end
if ~valid, return ; end
match = regexp(output, 'V(\d+\.\d+\.\d+)', 'match') ;
if isempty(match), valid = false ; return ; end
cuver = [1e4 1e2 1] * sscanf(match{1}, 'V%d.%d.%d') ;

% --------------------------------------------------------------------
function check_nvcc(cuda_root)
% --------------------------------------------------------------------
% Checks whether the nvcc is in the path. If not, guessed out of CudaRoot.
[status, ~] = system('nvcc --help');
if status ~= 0
  warning('nvcc not found in PATH. Trying to guess out of CudaRoot.');
  cuda_bin_path = fullfile(cuda_root, 'bin');
  prev_path = getenv('PATH');
  switch computer
    case 'PCWIN64', separator = ';';
    case {'GLNXA64', 'MACI64'}, separator = ':';
  end
  setenv('PATH', [prev_path separator cuda_bin_path]);
  [status, ~] = system('nvcc --help');
  if status ~= 0
    setenv('PATH', prev_path);
    error('Unable to find nvcc.');
  else
    fprintf('Location of nvcc (%s) added to your PATH.\n', cuda_bin_path);
  end
end

% --------------------------------------------------------------------
function cudaArch = get_cuda_arch(opts)
% --------------------------------------------------------------------
opts.verbose && fprintf('%s:\tCUDA: determining GPU compute capability (use the ''CudaArch'' option to override)\n', mfilename);
try
  gpu_device = gpuDevice();
  arch_code = strrep(gpu_device.ComputeCapability, '.', '');
  cudaArch = ...
      sprintf('-gencode=arch=compute_%s,code=\\\"sm_%s,compute_%s\\\" ', ...
              arch_code, arch_code, arch_code) ;
catch
  opts.verbose && fprintf(['%s:\tCUDA: cannot determine the capabilities of the installed GPU;'
                      'falling back to default\n'], mfilename);
  cudaArch = opts.defCudaArch;
end

