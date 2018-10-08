%%

close all;
clear all;
clc;

%%

run('~/GitHub/umuguc/matconvnet/matlab/vl_setupnn');

%%

rng(0);

[net, opts, imdb, info] = cnn_mnist_autoencoder;

save('net.mat', 'net', 'opts', 'info');

%%

% Training net:

%      layer|      1|      2|      3|      4|      5|      6|      7|      8|      9|     10|     11|     12|     13|
%       type|    cnv|sigmoid|    cnv|sigmoid|    cnv|    cnv|sigmoid|    cnv|sigmoid|    cnv|sigmoid|    cnv|sigmoidcrossentropyloss|
%    support|    1x1|    1x1|    1x1|    1x1|    1x1|    1x1|    1x1|    1x1|    1x1|    1x1|    1x1|    1x1|    1x1|
%     stride|      1|      1|      1|      1|      1|      1|      1|      1|      1|      1|      1|      1|      1|
%        pad|      0|      0|      0|      0|      0|      0|      0|      0|      0|      0|      0|      0|      0|
%    out dim|   1000|   1000|    500|    500|     30|    250|    250|    500|    500|   1000|   1000|    784|    784|
%   filt dim|    784|    n/a|   1000|    n/a|    250|     30|    n/a|    250|    n/a|    500|    n/a|   1000|    n/a|
% rec. field|      1|      1|      1|      1|      1|      1|      1|      1|      1|      1|      1|      1|      1|
% c/g net KB| 3066/0|    0/0| 1955/0|    0/0|   29/0|   30/0|    0/0|  490/0|    0/0| 1957/0|    0/0| 3066/0|    0/0|
% total network CPU/GPU memory: 10.3/0 MB

