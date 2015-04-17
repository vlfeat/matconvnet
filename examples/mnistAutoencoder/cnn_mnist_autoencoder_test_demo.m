%%

close all;
clear all;
clc;

%%

run('~/GitHub/umuguc/matconvnet/matlab/vl_setupnn');

%%

load('net.mat');
load(opts.imdbPath);

%%

N = [5 2];

Y = zeros(N(1) * N(2), 1);

h = figure;

for index = 1 : N(1) * N(2)
    
    im = imdb.images.data(:, :, :, end - index + 1);
    
    if opts.useGpu
        
        im = gpuArray(im);
        
    end
    
    subplot(N(1), 2 * N(2), 2 * index - 1);
    
    imagesc(reshape(im, 28, 28));
    
    axis off;
    axis square;
    
    drawnow;
    
    net.layers{end}.class = im;
    
    res = vl_simplenn(net, im, [], [], 'disableDropout', true);
    
    subplot(N(1), 2 * N(2), 2 * index);
    
    imagesc(reshape(res(end - 1).x, 28, 28));
    
    axis off;
    axis square;
    
    drawnow;
    
    Y(index) = gather(res(end).x);
    
end

disp(['Euclidean loss: ' num2str(mean(Y))]);

%%

% Test net:

%      layer|      1|      2|      3|      4|      5|      6|      7|      8|      9|     10|     11|     12|     13|     14|
%       type|    cnv|sigmoid|    cnv|sigmoid|    cnv|    cnv|sigmoid|    cnv|sigmoid|    cnv|sigmoid|    cnv|sigmoid|euclideanloss|
%    support|    1x1|    1x1|    1x1|    1x1|    1x1|    1x1|    1x1|    1x1|    1x1|    1x1|    1x1|    1x1|    1x1|    1x1|
%     stride|      1|      1|      1|      1|      1|      1|      1|      1|      1|      1|      1|      1|      1|      1|
%        pad|      0|      0|      0|      0|      0|      0|      0|      0|      0|      0|      0|      0|      0|      0|
%    out dim|   1000|   1000|    500|    500|     30|    250|    250|    500|    500|   1000|   1000|    784|    784|    784|
%   filt dim|    784|    n/a|   1000|    n/a|    250|     30|    n/a|    250|    n/a|    500|    n/a|   1000|    n/a|    n/a|
% rec. field|      1|      1|      1|      1|      1|      1|      1|      1|      1|      1|      1|      1|      1|      1|
% c/g net KB| 3066/0|    0/0| 1955/0|    0/0|   29/0|   30/0|    0/0|  490/0|    0/0| 1957/0|    0/0| 3066/0|    0/0|    0/0|
% total network CPU/GPU memory: 10.3/0 MB

