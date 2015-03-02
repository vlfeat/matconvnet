%% EXP batch normalisation

run ./matlab/vl_setupnn.m
addpath('examples');
%%
%net_chain = load('data/models/imagenet-caffe-alex.mat');

imdb = load('../dunravel/data/imdb_imagenet.mat');

%%

nd = 512;
ns = 256;
r = 30;
c = 30;

dtype = 'single'; range = 10;
%dtype = 'double'; range = 1;
grandn = @(q, varargin) gpuArray.randn(varargin{:});

grandn = @(q, varargin) randn(varargin{:});



q = RandStream('mt19937ar','Seed',1);
x = grandn(q, r, c, nd, ns, dtype) ;
g = grandn(q, nd, 1);
b = grandn(q, nd, 1);

fprintf('\n');
tic
[y] = vl_nnbnorm4(x,g,b) ;
toc

dzdy = grandn(q, size(y), dtype) ;
tic
[dzdx,dzdg,dzdb] = vl_nnbnorm4(x,g,b,dzdy) ;
toc
%
%tic
%[dzdx2,dzdg,dzdb] = vl_nnbnorm3(x,g,b,dzdy) ;
%toc

%%
vl_testder(@(x) vl_nnbnorm4(x,g,b), x, dzdy, dzdx, range * 1e-3) ;
vl_testder(@(g) vl_nnbnorm4(x,g,b), g, dzdy, dzdg, range * 1e-3) ;
vl_testder(@(b) vl_nnbnorm4(x,g,b), b, dzdy, dzdb, range * 1e-3) ;

%%
nd = 100;
dtype = 'single';
q = RandStream('mt19937ar','Seed',10);
x = randn(q, 1, 1, nd, 10, dtype) ;
y = vl_nnvar2(x) ;
dzdy = ones(1, 1, nd, dtype);
[dzdx] = vl_nnvar2(x, dzdy);

vl_testder(@(x) vl_nnvar2(x), x, dzdy, dzdx, 1e-3) ;
