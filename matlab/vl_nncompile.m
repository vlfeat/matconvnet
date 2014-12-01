function vl_nncompile( varargin )
run(fullfile(fileparts(mfilename('fullpath')), 'vl_setupnn.m'));

opts.enable_gpu = false;
opts.enable_imreadjpeg = true;
opts.debug = false;
opts.matlabroot = matlabroot();
opts.cudaroot = '';
opts.objext = 'obj';
opts.mexopts = {'-v','-largeArrayDims','-lmwblas'};
opts = vl_argparse(opts, varargin);

dest_dir = fullfile('matlab', 'mex');
src_dir = fullfile('matlab', 'src');

bits_src  = @(name) fullfile(src_dir, 'bits', [name '.cpp']);
bits_dst  = @(name) fullfile(src_dir, 'bits', [name '.' opts.objext]);
mex_src   = @(name) fullfile(src_dir, [name '.cpp']);
cumex_src = @(name) fullfile(src_dir, [name '.cu']);
mex_dst   = @(name) fullfile(dest_dir, [name '.' mexext]);

do.mex = @(src, dst, objfiles) ...
  mex(opts.mexopts{:}, src, '-output', dst, objfiles{:});
do.mexc = @(src, dst) [mex('-c', opts.mexopts{:}, src), ...
  movefile(objfilename(src), dst)];
do.cp = @(src, dst) copyfile(src, dst);
do.mv = @(src, dst) movefile(src, dst);
do.rm = @(dst) delete(dst);
do.mkdir = @(dst) mkdir(dst);

if ~opts.enable_gpu
  do.mkdir(dest_dir);
  obj_files = {'im2col', 'pooling', 'normalize', 'subsample'};
  cellfun(@(n) do.mexc(bits_src(n), bits_dst(n)), obj_files, ...
    'UniformOutput', false);
  obj_dsts = cellfun(bits_dst, obj_files, 'UniformOutput', false);
   
  mex_files = {'vl_nnconv', 'vl_nnpool', 'vl_nnnormalize'};
  cellfun(@(n) do.mex(mex_src(n), mex_dst(n), obj_dsts), mex_files, ...
    'UniformOutput', false);
end

  function fn = objfilename(src)
    [~, fn] = fileparts(src); fn = [fn '.' opts.objext];
  end

end
