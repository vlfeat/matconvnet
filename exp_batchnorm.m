%% EXP batch normalisation

run ./matlab/vl_setupnn.m
addpath('examples');

%net_chain = load('data/models/imagenet-caffe-alex.mat');

imdb = load('../dunravel/data/imdb_imagenet.mat');

%%
nd = 512;
ns = 256;
r = 20;
c = 20;
dtype = 'single'; range = 10;
%dtype = 'double'; range = 1;

q = RandStream('mt19937ar','Seed',1);
x = randn(q, r, c, nd, 10, dtype) ;
g = ones(nd, 1);
b = zeros(nd, 1);

tic
[y] = vl_nnbnorm3(x,g,b) ;
toc
%

dzdy = randn(q, size(y), dtype) ;
tic
[dzdx,dzdg,dzdb] = vl_nnbnorm3(x,g,b,dzdy) ;
toc
%%
vl_testder(@(x) vl_nnbnorm3(x,g,b), x, dzdy, dzdx, range * 1e-3) ;
vl_testder(@(g) vl_nnbnorm3(x,g,b), g, dzdy, dzdg, range * 1e-3) ;
vl_testder(@(b) vl_nnbnorm3(x,g,b), b, dzdy, dzdb, range * 1e-3) ;

%%
nd = 100;
dtype = 'single';
q = RandStream('mt19937ar','Seed',10);
x = randn(q, 1, 1, nd, 10, dtype) ;
y = vl_nnvar2(x) ;
dzdy = ones(1, 1, nd, dtype);
[dzdx] = vl_nnvar2(x, dzdy);

vl_testder(@(x) vl_nnvar2(x), x, dzdy, dzdx, 1e-3) ;
