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
%    below. Doing so requires the MATALB Parallel Computing
%    Toolbox. Furthermore, the 'nvcc' command must be in the
%    executable path or the MW_NVCC_PATH environment variable must
%    point to it (in MATLAB you can use SETENV() to set this
%    variable).
%
%    If you do not have 'nvcc', you can download it from NVIDIA as
%    part of the CUDA Toolkit. Please not that each MATLAB version
%    requires a particualr version of the CUDA devkit. The following
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
%    See also: VL_SETUPNN(),
%    http://mathworks.com/help/distcomp/run-mex-functions-containing-cuda-code.html

% Copyright (C) 2014 Karel Lenc and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

% Get MatConvNet root directory
root = fileparts(fileparts(mfilename('fullpath'))) ;
run(fullfile(root, 'matlab', 'vl_setupnn.m')) ;

opts.enableGpu = false ;
opts.verbose = false ;
opts.cudaRoot = [] ;
opts.debug = false ;
opts = vl_argparse(opts, varargin);
opts.enableGpu = opts.enableGpu || ~isempty(opts.cudaRoot) ;

% TODO debug mode
% TODO imreadjpeg

% Get MATLAB version as a number
mver = [1e4 1e2 1] * sscanf(version, '%d.%d.%d') ;

% Get architecture
switch computer
  case 'PCWIN64'
    arch = 'win64';
    if opts.enableGpu, cumex_libs{end+1} = '-lgpu' ; end
    objext = 'obj';
  case 'MACI64'
    arch = 'maci64';
    objext = 'o' ;
  case 'GLNXA64'
    arch = 'glnxa64' ;
    objext = 'o' ;
  otherwise
    error('Unsupported architecture %s. For UNIX-based systems try "!make".', computer) ;
end

% Create a directory for the MEX files if not already there
mex_dir = fullfile(root, 'matlab', 'mex') ;
if ~exist(mex_dir, 'dir'), mkdir(mex_dir) ; end

% Determine CUDA root directory if not specified
if opts.enableGpu && isempty(opts.cudaRoot)
  nvccPath = getenv('MW_NVCC_PATH') ;
  if isempty(nvccPath), nvccPath = 'nvcc' ; end
  nvccPath = whichc(nvccPath) ;  
  if ~exist(nvccPath, 'file')
    error('Could not find nvcc compiler (''%s''). Make sure that the MW_NVCC_PATH variable is set correctly.') ;
  end
  cupath = fileparts(fileparts(nvccPath)) ;
  switch computer
    case 'PCWIN64', cupath = fullfile(cupath, 'lib', 'x64') ;
    case 'MACI64', cupath = fullfile(cupath, 'lib') ;
    case 'GLNXA64', cupath = fullfile(cupath, 'lib64') ;
  end
end

% Setup environment to run NVCC to compile CUDA code if required
if opts.enableGpu
  setenv('MW_NVCC_PATH', nvccPath) ;
  if strcmp(arch, 'win64'), check_clpath() ; end
end

% The files to compile
cpp_src={fullfile(root, 'matlab', 'src', 'bits', 'im2col.cpp'), ...
         fullfile(root, 'matlab', 'src', 'bits', 'pooling.cpp'), ...
         fullfile(root, 'matlab', 'src', 'bits', 'normalize.cpp'), ...
         fullfile(root, 'matlab', 'src', 'bits', 'subsample.cpp')} ;

if ~opts.enableGpu
  mex_src={fullfile(root, 'matlab', 'src', 'vl_nnconv.cpp'), ...
           fullfile(root, 'matlab', 'src', 'vl_nnpool.cpp'), ...
           fullfile(root, 'matlab', 'src', 'vl_nnnormalize.cpp')} ;
else
  cpp_src={cpp_src{:}, ...
           fullfile(root, 'matlab', 'src', 'bits', 'im2col_gpu.cu'), ...
           fullfile(root, 'matlab', 'src', 'bits', 'pooling_gpu.cu'), ...
           fullfile(root, 'matlab', 'src', 'bits', 'normalize_gpu.cu'), ...
           fullfile(root, 'matlab', 'src', 'bits', 'subsample_gpu.cu')} ;
  mex_src={fullfile(root, 'matlab', 'src', 'vl_nnconv.cu'), ...
           fullfile(root, 'matlab', 'src', 'vl_nnpool.cu'), ...
           fullfile(root, 'matlab', 'src', 'vl_nnnormalize.cu')} ;
end

% Compiler options
mex_libs = {'-lmwblas'};
if opts.enableGpu
  cumex_libs = {} ;
  cumex_libs{end+1} = ['-L' fullfile(matlabroot, 'bin', arch)] ;
  cumex_libs{end+1} = ['-L' cupath] ;
  cumex_libs{end+1} = '-lcudart' ;
  cumex_libs{end+1} = '-lcublas' ;
end

if mver <= 80100, ext = 'sh' ; else ext = 'xml' ; end
mex_opts = {} ;
mex_opts{end+1} = '-largeArrayDims' ;
mex_opts{end+1} = '-f' ; 
mex_opts{end+1} = fullfile(root, 'matlab', 'src', 'config', ['mex_CUDA_' arch '.' ext]) ;
if opts.verbose, mex_opts{end+1} = '-v'; end
if opts.debug, mex_opts{end+1} = '-g' ; end

% Compile objects
tmp_dir = fullfile(mex_dir, '.build');
if ~exist(tmp_dir, 'dir'), mkdir(tmp_dir); end

cpp_dst = {} ;
for i=1:numel(cpp_src)
  [path,base,ext] = fileparts(cpp_src{i}) ;
  cpp_dst{i} = fullfile(tmp_dir, [base '.' objext]) ;
  mex('-c', mex_opts{:}, cpp_src{i}) ;
  movefile([base '.' objext], cpp_dst{i}) ;
end

% Compile MEX files
for i=1:numel(mex_src)
  [path,base,ext] = fileparts(mex_src{i}) ;
  dst = fullfile(mex_dir, [base '.' mexext]) ;
  switch ext
    case '.cpp', libs = mex_libs ;
    case '.cu', libs = horzcat(mex_libs, cumex_libs) ;
  end
  mex('-output', dst, mex_opts{:}, mex_src{i}, cpp_dst{:}, libs{:}) ;
end

% -------------------------------------------------------------------------
function p = whichc(cmd)
% -------------------------------------------------------------------------
switch computer
  case 'PCWIN64'
    [st, p] = system(sprintf('where %s', cmd));
  case {'MACI64', 'GLNXA64'}
    [st, p] = system(sprintf('which %s', cmd));
end
% TODO: is this supposed to strip whitespaces?
if st, p = nan;
else p = p(1:end-1); end;

% -------------------------------------------------------------------------
function check_clpath()
% -------------------------------------------------------------------------
% Checks whether the cl.ext is in the path (needed for the nvcc). If
% not, tries to guess the location out of mex configuration.
status = system('cl.exe -help');
if status == 1
  warning('CL.EXE not found in PATH. Trying to guess out of mex setup.');
  cc = mex.getCompilerConfigurations('c++');
  if isempty(cc)
    error('MEX is not configured. Run "mex -setup".');
  end
  prev_path = getenv('PATH');
  cl_path = fullfile(cc.Location, 'VC','bin','x86_amd64');
  setenv('PATH', [prev_path ';' cl_path]);
  status = system('cl.exe');
  if status == 1
    setenv('PATH', prev_path);
    error('Unable to find cl.exe');
  else
    fprintf('Location of cl.exe (%s) successfully added to PATH.\n', ...
      cl_path);
  end
end
