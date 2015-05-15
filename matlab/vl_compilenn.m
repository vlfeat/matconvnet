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
%       Set the verbosity level (0, 1 or 2).
%
%    `Debug`:: `false`
%       Set to true to compile the binaries with debugging
%       information.
%
%    `CudaMethod`:: Linux & Mac OS X: `mex`; Windows: `nvcc`
%       Choose the method used to compile the CUDA code. There are two
%       methods:
%
%       * The **`mex`** method uses the MATLAB MEX command with the
%         configuration file
%         `<MatConvNet>/matlab/src/config/mex_CUDA_<arch>.[sh/xml]`
%         This configuration file is in XML format since MATLAB 8.3
%         (R2014a) and is a Shell script for earlier versions. This
%         is, principle, the preferred method as it uses the
%         MATLAB-sanctioned compiler options.
%
%       * The **`nvcc`** method calls the NVIDIA CUDA compiler `nvcc`
%         directly to compile CUDA source code into object files.
%
%         This method allows to use a CUDA toolkit version that is not
%         the one that officially supported by a particular MATALB
%         version (see below). It is also the default method for
%         compilation under Windows and with CuDNN.
%
%    `CudaRoot`:: guessed automatically
%       This option specifies the path to the CUDA toolkit to use for
%       compilation.
%
%    `EnableImreadJpeg`:: `true`
%       Set this option to `true` to compile `vl_imreadjpeg`.
%
%    `ImageLibrary`:: `libjpeg` (Linux), `gdiplus` (Windows), `quartz` (Mac)
%       The image library to use for `vl_impreadjpeg`.
%
%    `ImageLibraryCompileFlags`:: platform dependent
%       A cell-array of additional flags to use when compiling
%       `vl_imreadjpeg`.
%
%    `ImageLibraryLinkFlags`:: platform dependent
%       A cell-array of additional flags to use when linking
%       `vl_imreadjpeg`.
%
%    `EnableCudnn`:: `false`
%       Set to `true` to compile CuDNN support.
%
%    `CudnnRoot`:: `'local/'`
%       Directory containing the unpacked binaries and header files of
%       the CuDNN library.
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
%    output of `gpuDevice()`) in order to be able to run the GPU code.
%    To change the compute capabilities, for `mex` `CudaMethod` edit
%    the particular config file.  For the 'nvcc' method, compute
%    capability is guessed based on the GPUDEVICE output. You can
%    override it by setting the 'CudaArch' parameter (e.g. in case of
%    multiple GPUs with various architectures).
%
%    See also: [Compliling MatConvNet](../install.md#compiling),
%    [Compiling MEX files containing CUDA
%    code](http://mathworks.com/help/distcomp/run-mex-functions-containing-cuda-code.html),
%    `vl_setup()`, `vl_imreadjpeg()`.

% Copyright (C) 2014-15 Karel Lenc and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

% Get MatConvNet root directory
root = fileparts(fileparts(mfilename('fullpath'))) ;
addpath(fullfile(root, 'matlab')) ;

% --------------------------------------------------------------------
%                                                        Parse options
% --------------------------------------------------------------------

opts.enableGpu        = false;
opts.enableImreadJpeg = true;
opts.enableCudnn      = false;
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
opts.cudnnRoot        = 'local' ;
opts = vl_argparse(opts, varargin);

% --------------------------------------------------------------------
%                                                     Files to compile
% --------------------------------------------------------------------

arch = computer('arch') ;
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
    case 'quartz', opts.imageLibraryLinkFlags = {'LDFLAGS=$LDFLAGS -framework Cocoa -framework ImageIO'} ;
    case 'gdiplus', opts.imageLibraryLinkFlags = {'-lgdiplus'} ;
  end
end

lib_src = {} ;
mex_src = {} ;

% Files that are compiled as CPP or CU depending on whether GPU support
% is enabled.
if opts.enableGpu, ext = 'cu' ; else, ext='cpp' ; end
lib_src{end+1} = fullfile(root,'matlab','src','bits',['data.' ext]) ;
lib_src{end+1} = fullfile(root,'matlab','src','bits',['datamex.' ext]) ;
lib_src{end+1} = fullfile(root,'matlab','src','bits',['nnconv.' ext]) ;
lib_src{end+1} = fullfile(root,'matlab','src','bits',['nnfullyconnected.' ext]) ;
lib_src{end+1} = fullfile(root,'matlab','src','bits',['nnsubsample.' ext]) ;
lib_src{end+1} = fullfile(root,'matlab','src','bits',['nnpooling.' ext]) ;
lib_src{end+1} = fullfile(root,'matlab','src','bits',['nnnormalize.' ext]) ;
lib_src{end+1} = fullfile(root,'matlab','src','bits',['nnbias.' ext]) ;
mex_src{end+1} = fullfile(root,'matlab','src',['vl_nnconv.' ext]) ;
mex_src{end+1} = fullfile(root,'matlab','src',['vl_nnconvt.' ext]) ;
mex_src{end+1} = fullfile(root,'matlab','src',['vl_nnpool.' ext]) ;
mex_src{end+1} = fullfile(root,'matlab','src',['vl_nnnormalize.' ext]) ;

% CPU-specific files
lib_src{end+1} = fullfile(root,'matlab','src','bits','impl','im2row_cpu.cpp') ;
lib_src{end+1} = fullfile(root,'matlab','src','bits','impl','subsample_cpu.cpp') ;
lib_src{end+1} = fullfile(root,'matlab','src','bits','impl','copy_cpu.cpp') ;
lib_src{end+1} = fullfile(root,'matlab','src','bits','impl','pooling_cpu.cpp') ;
lib_src{end+1} = fullfile(root,'matlab','src','bits','impl','normalize_cpu.cpp') ;
lib_src{end+1} = fullfile(root,'matlab','src','bits','impl','tinythread.cpp') ;

% GPU-specific files
if opts.enableGpu
  lib_src{end+1} = fullfile(root,'matlab','src','bits','impl','im2row_gpu.cu') ;
  lib_src{end+1} = fullfile(root,'matlab','src','bits','impl','subsample_gpu.cu') ;
  lib_src{end+1} = fullfile(root,'matlab','src','bits','impl','copy_gpu.cu') ;
  lib_src{end+1} = fullfile(root,'matlab','src','bits','impl','pooling_gpu.cu') ;
  lib_src{end+1} = fullfile(root,'matlab','src','bits','impl','normalize_gpu.cu') ;
  lib_src{end+1} = fullfile(root,'matlab','src','bits','datacu.cu') ;
end

% cuDNN-specific files
if opts.enableCudnn
  lib_src{end+1} = fullfile(root,'matlab','src','bits','impl','nnconv_cudnn.cu') ;
  lib_src{end+1} = fullfile(root,'matlab','src','bits','impl','nnbias_cudnn.cu') ;
  lib_src{end+1} = fullfile(root,'matlab','src','bits','impl','nnpooling_cudnn.cu') ;
end

% Other files
if opts.enableImreadJpeg
  mex_src{end+1} = fullfile(root,'matlab','src', ['vl_imreadjpeg.' ext]) ;
  lib_src{end+1} = fullfile(root,'matlab','src', 'bits', 'impl', ['imread_' opts.imageLibrary '.cpp']) ;
end

% --------------------------------------------------------------------
%                                                   Setup CUDA toolkit
% --------------------------------------------------------------------

if opts.enableGpu
  opts.verbose && fprintf('%s: * CUDA configuration *\n', mfilename) ;
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

  % CUDA arch string (select GPU architecture)
  if isempty(opts.cudaArch), opts.cudaArch = get_cuda_arch(opts) ; end
  opts.verbose && fprintf('%s:\tCUDA: NVCC architecture string: ''%s''.\n', ...
                          mfilename, opts.cudaArch) ;

  % Make sure NVCC is visible by MEX by setting the corresp. env. var
  if ~strcmp(getenv('MW_NVCC_PATH'), opts.nvccPath)
    warning('Setting the ''MW_NVCC_PATH'' environment variable to ''%s''', ...
            opts.nvccPath) ;
    setenv('MW_NVCC_PATH', opts.nvccPath) ;
  end
end

% --------------------------------------------------------------------
%                                                     Compiler options
% --------------------------------------------------------------------

% Build directories
mex_dir = fullfile(root, 'matlab', 'mex') ;
bld_dir = fullfile(mex_dir, '.build');
if ~exist(fullfile(bld_dir,'bits','impl'), 'dir')
  mkdir(fullfile(bld_dir,'bits','impl')) ;
end

% Compiler flags
flags.cc = {} ;
flags.link = {} ;
if opts.verbose > 1
  flags.cc{end+1} = '-v' ;
  flags.link{end+1} = '-v' ;
end
if opts.debug
  flags.cc{end+1} = '-g' ;
else
  flags.cc{end+1} = '-DNDEBUG' ;
end
if opts.enableGpu, flags.cc{end+1} = '-DENABLE_GPU' ; end
if opts.enableCudnn,
  flags.cc{end+1} = '-DENABLE_CUDNN' ;
  flags.cc{end+1} = ['-I' opts.cudnnRoot] ;
end
flags.link{end+1} = '-lmwblas' ;
switch arch
  case {'maci64', 'glnxa64'}
  case {'win64'}
    % VisualC does not pass this even if available in the CPU architecture
    flags.cc{end+1} = '-D__SSSE3__' ;
end

if opts.enableImreadJpeg
  flags.cc = horzcat(flags.cc, opts.imageLibraryCompileFlags) ;
  flags.link = horzcat(flags.link, opts.imageLibraryLinkFlags) ;
end

if opts.enableGpu
  flags.link{end+1} = ['-L' opts.cudaLibDir] ;
  flags.link{end+1} = '-lcudart' ;
  flags.link{end+1} = '-lcublas' ;
  switch arch
    case {'maci64', 'glnxa64'}
      flags.link{end+1} = '-lmwgpu' ;
    case 'win64'
      flags.link{end+1} = '-lgpu' ;
  end
  if opts.enableCudnn
    flags.link{end+1} = ['-L' opts.cudnnRoot] ;
    flags.link{end+1} = '-lcudnn' ;
  end
end

% For the MEX command
flags.link{end+1} = '-largeArrayDims' ;
flags.mexcc = flags.cc ;
flags.mexcc{end+1} = '-largeArrayDims' ;
flags.mexcc{end+1} = '-cxx' ;
if strcmp(arch, 'maci64')
  % CUDA prior to 7.0 on Mac require GCC libstdc++ instead of the native
  % Clang libc++. This should go away in the future.
  flags.mexcc{end+1} = 'CXXFLAGS=$CXXFLAGS -stdlib=libstdc++' ;
  flags.link{end+1} = 'LDFLAGS=$LDFLAGS -stdlib=libstdc++' ;
  if  ~verLessThan('matlab', '8.5.0')
    % Complicating matters, MATLAB 8.5.0 links to Clang c++ by default
    % when linking MEX files overriding the option above. More force
    % is needed:
    flags.link{end+1} = 'LINKLIBS=$LINKLIBS -L"$MATLABROOT/bin/maci64" -lmx -lmex -lmat -lstdc++' ;
  end
end
if opts.enableGpu
  flags.mexcu = flags.cc ;
  flags.mexcu{end+1} = '-largeArrayDims' ;
  flags.mexcu{end+1} = '-cxx' ;
  flags.mexcu(end+1:end+2) = {'-f' mex_cuda_config(root)} ;
  flags.mexcu{end+1} = ['NVCCFLAGS=' opts.cudaArch '$NVCC_FLAGS'] ;
end

% For the cudaMethod='nvcc'
if opts.enableGpu && strcmp(opts.cudaMethod,'nvcc')
  flags.nvcc = flags.cc ;
  flags.nvcc{end+1} = ['-I"' fullfile(matlabroot, 'extern', 'include') '"'] ;
  flags.nvcc{end+1} = ['-I"' fullfile(matlabroot, 'toolbox','distcomp','gpu','extern','include') '"'] ;
  if opts.debug
    flags.nvcc{end+1} = '-O0' ;
  end
  flags.nvcc{end+1} = '-Xcompiler' ;
  switch arch
    case {'maci64', 'glnxa64'}
      flags.nvcc{end+1} = '-fPIC' ;
    case 'win64'
      flags.nvcc{end+1} = '/MD' ;
      check_clpath(); % check whether cl.exe in path
  end
  flags.nvcc{end+1} = opts.cudaArch;
end

if opts.verbose
  fprintf('%s: * Compiler and linker configurations *\n', mfilename) ;
  fprintf('%s: \tintermediate build products directory: %s\n', mfilename, bld_dir) ;
  fprintf('%s: \tMEX files: %s/\n', mfilename, mex_dir) ;
  fprintf('%s: \tMEX compiler options: %s\n', mfilename, strjoin(flags.mexcc)) ;
  fprintf('%s: \tMEX linker options: %s\n', mfilename, strjoin(flags.link)) ;
end
if opts.verbose & opts.enableGpu
  fprintf('%s: \tMEX compiler options (CUDA): %s\n', mfilename, strjoin(flags.mexcu)) ;
end
if opts.verbose & opts.enableGpu & strcmp(opts.cudaMethod,'nvcc')
  fprintf('%s: \tNVCC compiler options: %s\n', mfilename, strjoin(flags.nvcc)) ;
end
if opts.verbose & opts.enableImreadJpeg
  fprintf('%s: * Reading images *\n', mfilename) ;
  fprintf('%s: \tvl_imreadjpeg enabled\n', mfilename) ;
  fprintf('%s: \timage library: %s\n', mfilename, opts.imageLibrary) ;
  fprintf('%s: \timage library compile flags: %s\n', mfilename, strjoin(opts.imageLibraryCompileFlags)) ;
  fprintf('%s: \timage library link flags: %s\n', mfilename, strjoin(opts.imageLibraryLinkFlags)) ;
end

% --------------------------------------------------------------------
%                                                              Compile
% --------------------------------------------------------------------

% Intermediate object files
srcs = horzcat(lib_src,mex_src) ;
parfor i = 1:numel(horzcat(lib_src, mex_src))
  [~,~,ext] = fileparts(srcs{i}) ; ext(1) = [] ;
  if strcmp(ext,'cu')
    if strcmp(opts.cudaMethod,'nvcc')
      nvcc_compile(opts, srcs{i}, toobj(bld_dir,srcs{i}), flags.nvcc) ;
    else
      mex_compile(opts, srcs{i}, toobj(bld_dir,srcs{i}), flags.mexcu) ;
    end
  else
    mex_compile(opts, srcs{i}, toobj(bld_dir,srcs{i}), flags.mexcc) ;
  end
end

% Link into MEX files
parfor i = 1:numel(mex_src)
  [~,base,~] = fileparts(mex_src{i}) ;
  objs = toobj(bld_dir, {mex_src{i}, lib_src{:}}) ;
  mex_link(opts, objs, mex_dir, flags.link) ;
end

% Reset path adding the mex subdirectory just created
vl_setupnn() ;

% --------------------------------------------------------------------
%                                                    Utility functions
% --------------------------------------------------------------------

% --------------------------------------------------------------------
function objs = toobj(bld_dir,srcs)
% --------------------------------------------------------------------
str = fullfile('matlab','src') ;
multiple = iscell(srcs) ;
if ~multiple, srcs = {srcs} ; end
for t = 1:numel(srcs)
  i = strfind(srcs{t},str);
  objs{t} = fullfile(bld_dir, srcs{t}(i+numel(str):end)) ;
end
if ~multiple, objs = objs{1} ; end
objs = strrep(objs,'.cpp',['.' objext]) ;
objs = strrep(objs,'.cu',['.' objext]) ;
objs = strrep(objs,'.c',['.' objext]) ;

% --------------------------------------------------------------------
function objs = mex_compile(opts, src, tgt, mex_opts)
% --------------------------------------------------------------------
mopts = {'-outdir', fileparts(tgt), src, '-c', mex_opts{:}} ;
opts.verbose && fprintf('%s: MEX CC: %s\n', mfilename, strjoin(mopts)) ;
mex(mopts{:}) ;

% --------------------------------------------------------------------
function obj = nvcc_compile(opts, src, tgt, nvcc_opts)
% --------------------------------------------------------------------
nvcc_path = fullfile(opts.cudaRoot, 'bin', 'nvcc');
nvcc_cmd = sprintf('"%s" -c "%s" %s -o "%s"', ...
                   nvcc_path, src, ...
                   strjoin(nvcc_opts), tgt);
opts.verbose && fprintf('%s: NVCC CC: %s\n', mfilename, nvcc_cmd) ;
status = system(nvcc_cmd);
if status, error('Command %s failed.', nvcc_cmd); end;

% --------------------------------------------------------------------
function mex_link(opts, objs, mex_dir, mex_flags)
% --------------------------------------------------------------------
mopts = {'-outdir', mex_dir, mex_flags{:}, objs{:}} ;
opts.verbose && fprintf('%s: MEX LINK: %s\n', mfilename, strjoin(mopts)) ;
mex(mopts{:}) ;

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
    paths = paths(strfind(paths, '.exe'));
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
  [status, output] = system(sprintf('"%s" --version', nvcc_path)) ;
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
  opts.verbose && fprintf(['%s:\tCUDA: cannot determine the capabilities of the installed GPU;' ...
                      'falling back to default\n'], mfilename);
  cudaArch = opts.defCudaArch;
end

