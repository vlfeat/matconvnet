function vl_compilenn( varargin )
% VL_COMPILENN  Compile the MatConvNet toolbox
%    The VL_COMPILENN() function compiles the binaries required to run
%    the MatConvNet toolbox. The function needs a properly configured
%    MEX compilation environment (see 'mex -setup').
%
%    VL_COMPILENN('OPTION', ARG, ...)  accepts the following options:
%
%    EnableGpu:: false
%       Set to true in order to enable GPU support.
%
%    enableImreadJpeg:: false
%       Set true to compile VL_IMREADJPEG. In order to succesfully compile,
%       libjpeg must be in linker search path. To adjust mex parameters,
%       see option 'imreadJpegFlags'.
%
%    imreadJpegFlags:: {'-ljpeg'}
%       Specify mex flags for vl_imreadjpeg compilation.
%
%    Verbose:: false
%       Set to true to turn on the verbose flag in the compiler.
%
%    Debug:: false
%       Set to true to compile the binaries with debugging
%       information.
%
%    COMPILING FOR GPU
%
%    In order to compile the GPU code use the 'EnableGpu' option
%    below. Doing so requires the MATALB Parallel Computing Toolbox.
%    For compilation the 'nvcc' command must be in the executable path.
%
%    If you do not have 'nvcc', you can download it from NVIDIA as
%    part of the CUDA Toolkit. Please note that each MATLAB version
%    requires a particular version of the CUDA devkit. The following
%    configurations have been tested successfully:
%
%    1) Windows 7 x64, MATLAB R2014a, Visual C++ 2010 and CUDA Toolkit
%       6.5 (unable to compile with Visual C++ 2013).
%    2) Windows 8 x64, MATLAB R2014a, Visual C++ 2013 and CUDA
%       Toolkit 6.5.
%    3) Mac OS X 10.9 and 10.10, MATLAB R2013a and R2013b, Xcode, CUDA
%       Toolkit 5.5.
%    4) GNU/Linux, MATALB R2014a, gcc, CUDA Toolkit 5.5.
%
%    If your system CUDA version is newer than the Matlab required version,
%    you can try compile CUDA object files directly with nvcc by setting
%    option 'cuObjMethod' to 'nvcc'.
%
%    CuObjMethod:: [Linux & Mac OSX : 'mex'] [Windows : 'nvcc']
%       Set the method used to compile CUDA objects. The 'mex' method
%       uses mex command with a particular mex configuration file
%       from <matconvnet_root>/matlab/src/config/mex_CUDA_<arch>.[sh/xml]
%       (xml configuration file used for Matlab version >= 8.3 (R2014a).
%       The 'nvcc' method directly calls nvidia cuda compiler. In some
%       cases with this method it is possible to compile GPU mex files with
%       newer CUDA toolkit than the required Matlab version.
%
%    With the 'nvcc' method, you can specify different CUDA root directory
%    (e.g. for compiling with different CUDA version) with the 'CudaRoot'
%    option.
%
%    Furthermore your GPU card must have ComputeCapability >= 2.0 (see
%    output of GPUDEVICE) in order to be able to run the GPU code.
%    To change the compute capabilities, for 'mex' 'CuObjMethod' edit
%    the particular config file.
%    For the 'nvcc' method, compute capability is guessed based on the
%    GPUDEVICE output. You can override it by setting the 'CudaArch'
%    parameter (e.g. in case of multiple GPUs with various architectures).
%
%    See also: VL_SETUPNN(), VL_IMREADJPEG(),
%    http://mathworks.com/help/distcomp/run-mex-functions-containing-cuda-code.html

% Copyright (C) 2014 Karel Lenc and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

% Get MatConvNet root directory
root = fileparts(fileparts(mfilename('fullpath'))) ;
run(fullfile(root, 'matlab', 'vl_setupnn.m')) ;

opts.enableGpu        = false;
opts.enableImreadJpeg = false;
opts.imreadJpegFlags  = {'-ljpeg'};
opts.verbose          = false;
opts.debug            = false;
opts.cudaRoot         = [];
opts.cudaArch         = [];
opts.defCudaArch      = [...
  '-gencode=arch=compute_20,code=\"sm_20,compute_20\" '...
  '-gencode=arch=compute_30,code=\"sm_30,compute_30\"'];
opts.cuObjMethod = guess_cuobj_method();
opts = vl_argparse(opts, varargin);

if opts.enableGpu
  if isempty(opts.cudaRoot), opts.cudaRoot = guess_cuda_root(); end;
  if isempty(opts.cudaRoot), error('nvcc or ''cudaRoot'' not found.'); end;
  check_nvcc(opts.cudaRoot);
end
% TODO imreadjpeg

% The files to compile
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

% Mex arguments
mex_libs = {'-lmwblas'};
cumex_libs = {['-L' matlab_libdir], ['-L' cuda_libdir(opts.cudaRoot)], ...
  '-lcudart', '-lcufft', '-lcublas'};
if strcmp(computer, 'PCWIN64'), cumex_libs{end+1} = '-lgpu'; end;

mex_opts = {'-largeArrayDims'};
if opts.verbose, mex_opts{end+1} = '-v'; end
if opts.debug, mex_opts{end+1} = '-g' ; end

% Create a directory for the MEX files if not already there
mex_dir = fullfile(root, 'matlab', 'mex') ;
if ~exist(mex_dir, 'dir'), mkdir(mex_dir) ; end
tmp_dir = fullfile(mex_dir, '.build');
if ~exist(tmp_dir, 'dir'), mkdir(tmp_dir); end

% Compile
obj_files = cpp_compile(cpp_src, tmp_dir, mex_opts);
if ~opts.enableGpu
  mex_link(mex_src, obj_files, mex_libs, mex_dir, mex_opts);
else
  switch opts.cuObjMethod
    case 'mex'
      mex_opts = [mex_opts {'-f' mex_cuda_config(root)}];
      cuobj_files = cpp_compile(cu_src, tmp_dir, mex_opts);
      mss = mex_cu_src;
    case 'nvcc'
      mex_opts = [mex_opts {'-cxx'}];
      cuobj_files = nvcc_compile(cu_src, tmp_dir, opts);
      mss =  nvcc_compile(mex_cu_src, tmp_dir, opts);
  end
  mex_link(mss, [obj_files cuobj_files], [mex_libs cumex_libs], ...
    mex_dir, mex_opts);
end

if opts.enableImreadJpeg
  imr_src = fullfile(root, 'matlab', 'src', 'vl_imreadjpeg.c');
  mex_link({imr_src}, {}, [mex_libs opts.imreadJpegFlags], mex_dir, mex_opts);
end

% -------------------------------------------------------------------------
function objs = cpp_compile(srcs, tmp_dir, mex_opts)
% -------------------------------------------------------------------------
% Compile objects
objs = cell(1, numel(srcs)) ;
for i=1:numel(srcs)
  [~,base] = fileparts(srcs{i}) ;
  objs{i} = fullfile(tmp_dir, [base '.' objext]) ;
  mex('-c', mex_opts{:}, srcs{i}) ;
  movefile([base '.' objext], objs{i}) ;
end

% -------------------------------------------------------------------------
function objs = nvcc_compile(srcs, tmp_dir, opts)
% -------------------------------------------------------------------------
% Compile CUDA objects with nvcc
nvcc_path = fullfile(opts.cudaRoot, 'bin', 'nvcc');

% Guess CUDA arch
if isempty(opts.cudaArch)
  fprintf('Guessing GPU compute capability...\n');
  try
    gpu_device = gpuDevice();
      arch_code = strrep(gpu_device, '.', '');
    opts.cudaArch = ...
      sprintf('-gencode=arch=compute_%d,code=\"sm_%d,compute_%d\" ', ...
      repmat(arch_code, 1, 3));
  catch
    warning('Unable to evaluate gpuDevice(). Using default CUDA arch.');
    opts.cudaArch = opts.defCudaArch;
  end

  fprintf('nvcc arch set to %s. ', opts.cudaArch);
  fprintf('Can be changed with ''CudaArch'' option.\n');
end

% nvcc options
nvcc_opts = [opts.cudaArch, ' -DenableGpu' ...
  ' -I"' fullfile(matlabroot, 'extern','include') '"' ...
  ' -I"', fullfile(matlabroot, 'toolbox','distcomp','gpu','extern','include') '"'];
if opts.verbose, nvcc_opts = [nvcc_opts ' -v']; end
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

objs = cell(1, numel(srcs));
for i=1:numel(srcs)
  [~,base] = fileparts(srcs{i}) ;
  objs{i} = fullfile(tmp_dir, [base '.' objext]) ;
  nvcc_cmd = sprintf('"%s" -c "%s" %s -o "%s"', nvcc_path, srcs{i}, ...
    nvcc_opts, objs{i});
  if opts.verbose, fprintf('Running: %s\n', nvcc_cmd); end
  status = system(nvcc_cmd);
  if status, error('Command %s failed.', nvcc_cmd); end;
end

% -------------------------------------------------------------------------
function mex_link(srcs, objs, libs, mex_dir, mex_opts)
% -------------------------------------------------------------------------
% Link MEX files
for i=1:numel(srcs)
  [~,base] = fileparts(srcs{i}) ;
  dst = fullfile(mex_dir, [base '.' mexext]) ;
  mex('-output', dst, mex_opts{:}, srcs{i}, objs{:}, libs{:}) ;
end

% -------------------------------------------------------------------------
function ext = objext()
% -------------------------------------------------------------------------
% Get object extension
switch computer
  case 'PCWIN64', ext = 'obj';
  case {'MACI64', 'GLNXA64'}, ext = 'o' ;
  otherwise, error('Unsupported architecture %s.', computer) ;
end

% -------------------------------------------------------------------------
function libpath = matlab_libdir()
% -------------------------------------------------------------------------
% Get matlab libraries path
switch computer
  case 'PCWIN64', arch = 'win64';
  case 'MACI64',  arch = 'maci64';
  case 'GLNXA64', arch = 'glnxa64' ;
  otherwise, error('Unsupported architecture %s.', computer) ;
end
libpath = fullfile(matlabroot, 'bin', arch);

% -------------------------------------------------------------------------
function libpath = cuda_libdir(cuda_root)
% -------------------------------------------------------------------------
% Get CUDA libraries path
switch computer
  case 'PCWIN64', libpath = fullfile(cuda_root, 'lib', 'x64') ;
  case 'MACI64',  libpath = fullfile(cuda_root, 'lib') ;
  case 'GLNXA64', libpath = fullfile(cuda_root, 'lib64') ;
  otherwise, error('Unsupported architecture %s.', computer) ;
end

% -------------------------------------------------------------------------
function conf_file = mex_cuda_config(root)
% -------------------------------------------------------------------------
% Get mex CUDA config file
mver = [1e4 1e2 1] * sscanf(version, '%d.%d.%d') ;
if mver <= 80200, ext = 'sh' ; else ext = 'xml' ; end
arch = lower(computer);
config_dir = fullfile(root, 'matlab', 'src', 'config');
conf_file = fullfile(config_dir, ['mex_CUDA_' arch '.' ext]);
fprintf('CUDA mex config file: "%s"\n', conf_file);

% -------------------------------------------------------------------------
function check_clpath()
% -------------------------------------------------------------------------
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
function cuda_root = guess_cuda_root()
% -------------------------------------------------------------------------
% Guess the cuda root path
switch computer
  case 'PCWIN64'
    def_path = 'c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5';
    [~, nvcc_path] = system('where nvcc.exe');
    nvcc_path = strtrim(nvcc_path);
    % On some systems, 'where' can return multiple paths
    num_found = numel(strfind(strtrim(nvcc_path), '.exe'));
    if num_found > 1, nvcc_path = ''; end;
  case 'MACI64'
    def_path = '/Developer/NVIDIA/CUDA-5.5/';
    [~, nvcc_path] = system('which nvcc');
    nvcc_path = strtrim(nvcc_path);
  case 'GLNXA64'
    def_path = '/usr/local/cuda/';
    [~, nvcc_path] = system('which nvcc');
    nvcc_path = strtrim(nvcc_path);
  otherwise
    error('Unsupported architecture %s.', computer) ;
end

if exist(nvcc_path, 'file')
  cuda_root = fileparts(fileparts(nvcc_path));
else
  if exist(def_path, 'dir')
    cuda_root = def_path;
  else
    cuda_root = '';
    return;
  end
end
fprintf('CUDA root guessed as: "%s", can be changed with ''cudaRoot'' option.\n', ...
  cuda_root);

% -------------------------------------------------------------------------
function check_nvcc(cuda_root)
% -------------------------------------------------------------------------
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

% -------------------------------------------------------------------------
function method = guess_cuobj_method()
% -------------------------------------------------------------------------
% Guess cuda compilation method
switch computer
  case 'PCWIN64', method = 'nvcc';
  case 'MACI64',  method = 'mex';
  case 'GLNXA64', method = 'mex';
  otherwise, error('Unsupported architecture %s.', computer) ;
end