function vl_compilenn( varargin )
% VL_COMPILENN  Compile the MatConvNet toolbox
%    VL_COMPILENN() function compiles the MatConvNet toolbox.
%    Simple compilation tool for Windows 64 platforms without need for Make
%    implementation.
%    This function needs properly configured mex compilation environment
%    (see 'mex -setup').
%    For GPU code, 'nvcc' must be in PATH.
%    GPU compilation tested on configurations: 
%      Windows 7 x64, Matlab R2014a, Visual C++ 2010 and CUDA Toolkit 6.5.
%        (unable to compile with Visual C++ 2013).
%      Windows 8 x64, Matlab R2014a, Visual C++ 2013 and CUDA Tollkit 6.5.
%    
%  VL_COMPILENN accepts the following options:
%  enable_gpu :
%    When true, enables GPU support.
%
%  verbose :
%    When true, set the verbose flags for the compilers.
%  
%  In case of GPU compilation, following options are also used:
%  cuda_path : 
%    CUDA toolkit path (root toolkit path). By default set to:
%    'c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5'
%
%  cuda_arch : 
%    NVCC arguments specifying the target GPU architecture. By default, set
%    to include PTX code and binaries for compute capabilities 2.0 and 3.0.
%    See NVCC documentation for details.

% Copyright (C) 2014 Karel Lenc and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
dest_dir = fullfile('matlab', 'mex');
if ~exist(dest_dir, 'dir'), mkdir(dest_dir); end
run(fullfile(fileparts(mfilename('fullpath')), 'vl_setupnn.m'));

opts.enable_gpu = false;
opts.verbose = false;
opts.cuda_path = 'c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5';
opts.cuda_arch = ['-gencode=arch=compute_20,code=\"sm_20,compute_20\" '...
  '-gencode=arch=compute_30,code=\"sm_30,compute_30\"'];
opts = vl_argparse(opts, varargin);

% TODO debug mode
% TODO imreadjpeg

% Libraries
mex_libs = {'-largeArrayDims','-lmwblas'};
cumex_libs = {'-lcudart', '-lcufft', '-lcublas', '-lgpu'};

% OS specific options
nvcc_opts = '';
mex_opts = {};
switch computer
  case 'PCWIN64'
    arch = 'win64';
    culib_path = 'x64';
    nvcc_command = 'nvcc.exe';
    nvcc_opts = [nvcc_opts '-Xcompiler  /MD ']; % Compile as dynamic lib
    objext = 'obj';
    check_clpath();
  otherwise
    error('Unsupported platform. For Linux and Mac OS X please use "!make".');
end

% CUDA options
if opts.enable_gpu
  if ~exist(opts.cuda_path, 'dir')
    error('Invalid CUDA path'); 
  end
  
  culib_path = fullfile(opts.cuda_path, 'lib', culib_path);
  if ~exist(culib_path, 'dir')
    error('CUDA lib dir %s does not exist.', culib_path); 
  end;
  
  nvcc_path = fullfile(opts.cuda_path, 'bin', nvcc_command);
  if ~exist(nvcc_path, 'file')
    error('NVCC not found.'); 
  end;
  
  % Compiler options
  cumex_opts = {['-L/MATLAB_ROOT/bin/' arch], ['-L' culib_path]};
  nvcc_opts = [nvcc_opts , ...
    opts.cuda_arch...
    ' -DENABLE_GPU'...
    ' -I"' fullfile(matlabroot, 'extern','include') '"'...
    ' -I"', fullfile(matlabroot, 'toolbox','distcomp','gpu','extern','include') '"'];
  if opts.verbose
    nvcc_opts = ['-v ' nvcc_opts];
  end
end;

if opts.verbose
  mex_opts{end+1} = '-v'; 
end

% Set directories
src_dir = fullfile('matlab', 'src');
tmp_dir = fullfile('matlab', 'mex', '.build');
if ~exist(tmp_dir, 'dir'), mkdir(tmp_dir); end

bits_src  = @(name) fullfile(src_dir, 'bits', [name '.cpp']);
bits_dst  = @(name) fullfile(tmp_dir, [name '.' objext]);
mex_src   = @(name) fullfile(src_dir, [name '.cpp']);
cumex_src = @(name) fullfile(src_dir, [name '.cu']);
cubits_src  = @(name) fullfile(src_dir, 'bits', [name '.cu']);
cubits_dst = @(name) fullfile(tmp_dir, [name '.o']);
mex_dst   = @(name) fullfile(dest_dir, [name '.' mexext]);

% Definitions of compile operations
do.mex = @(src, dst, objfiles) ...
  mex(mex_opts{:}, mex_libs{:}, src, '-output', dst, objfiles{:});
do.mexc = @(src, dst) [mex('-c', src, mex_opts{:}), ...
  movefile(objfilename(src), dst)];
if opts.enable_gpu
  do.cumex = @(src, dst, objfiles) ...
    mex(mex_opts{:}, cumex_opts{:}, mex_libs{:}, cumex_libs{:}, src, ...
    '-output', dst, objfiles{:});
  do.cumexc = @(src, dst, objfiles) ...
    systemc(sprintf('"%s" -O3 -DNDEBUG -c %s %s -o %s', ...
    nvcc_path, src, nvcc_opts, dst));
end

% Compile the common CPU objects
bits_files = {'im2col', 'pooling', 'normalize', 'subsample'};
cellfun(@(n) do.mexc(bits_src(n), bits_dst(n)), bits_files, ...
  'UniformOutput', false);
bits_dsts = cellfun(bits_dst, bits_files, 'UniformOutput', false);

if ~opts.enable_gpu
  % Compile CPU only mex files
  mex_files = {'vl_nnconv', 'vl_nnpool', 'vl_nnnormalize'};
  cellfun(@(n) do.mex(mex_src(n), mex_dst(n), bits_dsts), mex_files, ...
    'UniformOutput', false);
else
  % Compile GPU common objects
  cubits_files = ...
    {'im2col_gpu', 'pooling_gpu', 'normalize_gpu', 'subsample_gpu'};
  cellfun(@(n) do.cumexc(cubits_src(n), cubits_dst(n)), cubits_files, ...
    'UniformOutput', false);
  cubits_dsts = cellfun(cubits_dst, cubits_files, 'UniformOutput', false);
  
  % Compile GPU mex files
  mex_files = {'vl_nnconv', 'vl_nnpool', 'vl_nnnormalize'};
  for mi = 1:numel(mex_files)
    src_f = cumex_src(mex_files{mi});
    obj_f = cubits_dst(mex_files{mi});
    mex_f = mex_dst(mex_files{mi});
    do.cumexc(src_f, obj_f);
    do.cumex(obj_f, mex_f, [cubits_dsts bits_dsts]);
  end
end

  function [status] = systemc(cmd)
    [status] = system(cmd);
    if status, error('%s failed.', cmd); end
  end

  function fn = objfilename(src)
    [~, fn] = fileparts(src); fn = [fn '.' objext];
  end

  function check_clpath()
    % Checks whether the cl.ext is in the path (needed for the nvcc). If
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
  end
end
