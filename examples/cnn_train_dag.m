function [net,stats] = cnn_train_dag(net, imdb, getBatch, varargin)
%CNN_TRAIN_DAG Demonstrates training a CNN using the DagNN wrapper
%    CNN_TRAIN_DAG() is similar to CNN_TRAIN(), but works with
%    the DagNN wrapper instead of the SimpleNN wrapper.

% Copyright (C) 2014-16 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

% Hacked by Ligong Han
% Apr 14, 2017
% Apr 30, 2017 sampler
% May 01, 2017 continue
% Jun 10, 2017 train val test, batch
% Jun 27, 2017 rng for GPU
% Jul 03, 2017 requires_grad from PyTorch

opts.expDir = fullfile('data','exp');
opts.continue = true; % whether of nor last checkpoint overrides net
opts.startEpoch = []; % starts from 1
opts.batchSize = 256;
opts.numSubBatches = 1;
opts.train = [];
opts.val = [];
opts.test = [];
opts.gpus = [];
opts.prefetch = false;
opts.epochSize = inf;
opts.numEpochs = 300;
opts.saveEpochs = [];
opts.learningRate = 0.001;
opts.weightDecay = 0.0005;
opts.sampler = []; % sampler(imdb, index), default: @(~, i, varargin) i
opts.solver = [];  % Empty array means use the default SGD solver
[opts, varargin] = vl_argparse(opts, varargin);
if ~isempty(opts.solver)
    assert(isa(opts.solver, 'function_handle') && nargout(opts.solver) == 2, ...
        'Invalid solver; expected a function handle with two outputs.');
    % Call without input arguments, to get default options
    opts.solverOpts = opts.solver();
end
opts.momentum = 0.9;
opts.saveSolverState = true;
opts.nesterovUpdate = false;
opts.randomSeed = 0;
opts.profile = false;
opts.parameterServer.method = 'mmap';
opts.parameterServer.prefix = 'mcn';

opts.derOutputs = {'objective', 1};
opts.extractStatsFn = @extractStats;
opts.plotStatistics = true;
opts.postEpochFn = [];  % postEpochFn(net, params, state) called after each epoch;
                        % can return a new learning rate, 0 to stop, [] for no change
[opts, ~] = vl_argparse(opts, varargin);

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir); end
if isempty(opts.sampler), opts.sampler = @(~, ids, varargin) ids; end
if isempty(opts.saveEpochs), opts.saveEpochs = 1:opts.numEpochs; end
if isempty(opts.train), opts.train = find(imdb.images.set == 1); end
if isempty(opts.val), opts.val = find(imdb.images.set == 2); end
if isempty(opts.test), opts.test = find(imdb.images.set == 3); end
if isscalar(opts.train) && isnumeric(opts.train) && isnan(opts.train), opts.train = []; end
if isscalar(opts.val) && isnumeric(opts.val) && isnan(opts.val), opts.val = []; end
if isscalar(opts.test) && isnumeric(opts.test) && isnan(opts.test), opts.test = []; end
opts.train = opts.train(:); opts.val = opts.val(:); opts.test = opts.test(:); % column vector

% -------------------------------------------------------------------------
%                                                            Initialization
% -------------------------------------------------------------------------

evaluateMode = isempty(opts.train);
if ~evaluateMode
    if isempty(opts.derOutputs)
        error('DEROUTPUTS must be specified when training.\n');
    end
end

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
modelFigPath = fullfile(opts.expDir, 'net-train.pdf');

% % start = opts.continue * findLastCheckpoint(opts.expDir);
start = findLastCheckpoint(opts.expDir);
if ~isempty(opts.startEpoch), start = opts.startEpoch-1; end
if start >= opts.numEpochs, stats = []; return; end
if ~opts.continue
    % unpack net
    if isa(net, 'dagnn.DagNN')
        fprintf('%s: loading net without stats, starting from epoch %d\n', ...
            mfilename, start+1);
        state = [];
    elseif isa(net, 'struct') || isa(net, 'cell')
        % net can still be a DagNN object
        S = struct('net', [], 'state', [], 'stats', []);
        [S, ~] = vl_argparse(S, net);
        [net, state, stats] = deal(dagnn.DagNN.loadobj(S.net), S.state, S.stats);
        fprintf('%s: loading model, starting from epoch %d\n', mfilename, start+1);
    else
        error('NET can only be dagnn.DagNN, struct, or cell array.\n');
    end
else
    if start >= 1
        % load last checkpoint
        fprintf('%s: resuming by loading epoch %d\n', mfilename, start);
        [net, state, stats] = loadState(modelPath(start));
    else
        state = [];
    end
end
if ~exist('stats', 'var'), stats = struct('train', [], 'val', [], 'test', []); end

for epoch = start+1:opts.numEpochs
    
    % Set the random seed based on the epoch and opts.randomSeed.
    % This is important for reproducibility, including when training
    % is restarted from a checkpoint.
    
    rng(epoch + opts.randomSeed);
    parallel.gpu.rng(epoch + opts.randomSeed);
    prepareGPUs(opts, epoch == start+1);
    
    % Train for one epoch.
    params = opts;
    params.epoch = epoch;
    params.imdb = imdb;
    params.getBatch = getBatch;
    params.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate)));
    params.momentum = opts.momentum(min(epoch, numel(opts.momentum)));
    % % params.train = opts.train(randperm(numel(opts.train))); % shuffle
    % % params.train = params.train(1:min(opts.epochSize, numel(opts.train)));
    % % params.val = opts.val(randperm(numel(opts.val)));
    
    % opts.train is initially forced to be a column vector; after sampling,
    % it's a N-by-M matrix where N is the number of images/samples and M is
    % the number of datasets.
    params.train = opts.sampler(imdb, opts.train, 'mode', 'train'); % sample
    params.train = params.train(randperm(size(params.train,1)),:); % shuffle
    params.train = params.train(1:min(opts.epochSize,size(params.train,1)),:);
    params.val = opts.sampler(imdb, opts.val, 'mode', 'val'); % no need to shuffle
    params.test = opts.sampler(imdb, opts.test, 'mode', 'test');
    
    if numel(opts.gpus) <= 1
        [net, state] = processEpoch(net, state, params, 'train');
        [net, state] = processEpoch(net, state, params, 'val');
        [net, state] = processEpoch(net, state, params, 'test');
        if ~evaluateMode && ismember(epoch, opts.saveEpochs)
            saveState(modelPath(epoch), net, state);
        end
        lastStats = state.stats;
    else
        spmd
            [net, state] = processEpoch(net, state, params, 'train');
            [net, state] = processEpoch(net, state, params, 'val');
            [net, state] = processEpoch(net, state, params, 'test');
            if labindex == 1 && ~evaluateMode && ismember(epoch, opts.saveEpochs)
                saveState(modelPath(epoch), net, state);
            end
            lastStats = state.stats;
        end
        lastStats = accumulateStats(lastStats);
    end
    
    % % stats.train(epoch) = lastStats.train;
    % % stats.val(epoch) = lastStats.val;
    stats.train = appendElement(stats.train, lastStats.train, epoch);
    stats.val = appendElement(stats.val, lastStats.val, epoch);
    stats.test = appendElement(stats.test, lastStats.test, epoch);
    clear lastStats;
    % % saveStats(modelPath(epoch), stats);
    if ismember(epoch, opts.saveEpochs), saveStats(modelPath(epoch), stats); end
    
    if opts.plotStatistics
        switchFigure(1); clf;
        plots = setdiff(cat(2, ...
            fieldnames(stats.train)', ...
            fieldnames(stats.val)', ...
            fieldnames(stats.test)'), {'num', 'time'}, 'stable');
        for p = plots
            p_ = char(p);
            values = zeros(0, epoch);
            leg = {};
            for f = {'train', 'val', 'test'}
                f_ = char(f);
                if isfield(stats.(f_), p_)
                    tmp = [stats.(f_).(p_)];
                    values(end+1,:) = tmp(1,:)'; %#ok<AGROW>
                    leg{end+1} = f_; %#ok<AGROW>
                end
            end
            subplot(1, numel(plots), find(strcmp(p_,plots)));
            plot(1:epoch, values', 'o-');
            xlabel('epoch');
            title(strrep(p_, '_', '-'));
            legend(leg{:}, 'Location', 'Southwest');
            grid on;
        end
        drawnow;
        % % print(1, modelFigPath, '-dpdf');
        try
            set(gcf, 'PaperOrientation', 'landscape');
            print(1, modelFigPath, '-bestfit', '-dpdf');
            savefig(gcf, strrep(modelFigPath, '.pdf', '.fig'));
        catch
            warning('Cannot save figures.');
        end
    end
    
    if ~isempty(opts.postEpochFn)
        if nargout(opts.postEpochFn) == 0
            opts.postEpochFn(net, params, state);
        else
            lr = opts.postEpochFn(net, params, state);
            if ~isempty(lr), opts.learningRate = lr; end
            if opts.learningRate == 0, break; end
        end
    end
end

% With multiple GPUs, return one copy
if isa(net, 'Composite'), net = net{1}; end

% -------------------------------------------------------------------------
function [net, state] = processEpoch(net, state, params, mode)
% -------------------------------------------------------------------------
% Note that net is not strictly needed as an output argument as net
% is a handle class. However, this fixes some aliasing issue in the
% spmd caller.

% initialize with momentum 0
if isempty(state) || isempty(state.solverState)
    state.solverState = cell(1, numel(net.params));
    state.solverState(:) = {0};
end

% move CNN  to GPU as needed
numGpus = numel(params.gpus);
if numGpus >= 1
    net.move('gpu');
    for i = 1:numel(state.solverState)
        s = state.solverState{i};
        if isnumeric(s)
            state.solverState{i} = gpuArray(s);
        elseif isstruct(s)
            state.solverState{i} = structfun(@gpuArray, s, 'UniformOutput', false);
        end
    end
end
if numGpus > 1
    parserv = ParameterServer(params.parameterServer);
    net.setParameterServer(parserv);
else
    parserv = [];
end

% profile
if params.profile
    if numGpus <= 1
        profile clear;
        profile on;
    else
        mpiprofile reset;
        mpiprofile on;
    end
end

num = 0;
epoch = params.epoch;
subset = params.(mode);
adjustTime = 0;

stats.num = 0; % return something even if subset = []
stats.time = 0;

start = tic;
for t = 1:params.batchSize:size(subset,1)
    fprintf('%s: epoch %02d: %3d/%3d:', mode, epoch, ...
        fix((t-1)/params.batchSize)+1, ceil(size(subset,1)/params.batchSize));
    batchSize = min(params.batchSize, size(subset,1) - t + 1);
    
    for s = 1:params.numSubBatches
        % get this image batch and prefetch the next
        batchStart = t + (labindex-1) + (s-1) * numlabs;
        batchEnd = min(t+params.batchSize-1, size(subset,1));
        batch = subset(batchStart : params.numSubBatches * numlabs : batchEnd,:);
        num = num + size(batch,1);
        if size(batch,1) == 0, continue; end
        
        inputs = params.getBatch(params.imdb, batch, 'mode', mode, 'epoch', epoch);
        
        if params.prefetch
            if s == params.numSubBatches
                batchStart = t + (labindex-1) + params.batchSize;
                batchEnd = min(t+2*params.batchSize-1, size(subset,1));
            else
                batchStart = batchStart + numlabs;
            end
            nextBatch = subset(batchStart : params.numSubBatches * numlabs : batchEnd,:);
            params.getBatch(params.imdb, nextBatch, 'mode', mode, 'epoch', epoch);
        end
        
        if strcmp(mode, 'train')
            net.mode = 'normal';
            net.accumulateParamDers = (s ~= 1);
            net.eval(inputs, params.derOutputs, 'holdOn', s < params.numSubBatches);
        else
            net.mode = 'test';
            net.eval(inputs);
        end
    end
    
    % Accumulate gradient.
    if strcmp(mode, 'train')
        if ~isempty(parserv), parserv.sync(); end
        state = accumulateGradients(net, state, params, batchSize, parserv);
    end
    
    % Get statistics.
    time = toc(start) + adjustTime;
    batchTime = time - stats.time;
    stats.num = num;
    stats.time = time;
    stats = params.extractStatsFn(stats, net);
    currentSpeed = batchSize / batchTime;
    averageSpeed = (t + batchSize - 1) / time;
    if t == 3*params.batchSize + 1
        % compensate for the first three iterations, which are outliers
        adjustTime = 4*batchTime - time;
        stats.time = time + adjustTime;
    end
    
    fprintf(' %.1f (%.1f) Hz', averageSpeed, currentSpeed);
    for f = setdiff(fieldnames(stats)', {'num', 'time'}, 'stable')
        f_ = char(f);
        fprintf(' %s: %.3f', f_, stats.(f_));
    end
    fprintf('\n');
end

% Save back to state.
state.stats.(mode) = stats;
if params.profile
    if numGpus <= 1
        state.prof.(mode) = profile('info');
        profile off;
    else
        state.prof.(mode) = mpiprofile('info');
        mpiprofile off;
    end
end
if ~params.saveSolverState
    state.solverState = [];
else
    for i = 1:numel(state.solverState)
        s = state.solverState{i};
        if isnumeric(s)
            state.solverState{i} = gather(s);
        elseif isstruct(s)
            state.solverState{i} = structfun(@gather, s, 'UniformOutput', false);
        end
    end
end

net.reset();
net.move('cpu');

% -------------------------------------------------------------------------
function state = accumulateGradients(net, state, params, batchSize, parserv)
% -------------------------------------------------------------------------
numGpus = numel(params.gpus);
otherGpus = setdiff(1:numGpus, labindex); %#ok

for p = 1:numel(net.params)
    
    if ~net.params(p).requiresGrad, continue; end
    
    if ~isempty(parserv)
        parDer = parserv.pullWithIndex(p);
    else
        parDer = net.params(p).der;
    end
    
    switch net.params(p).trainMethod
        case 'average' % mainly for batch normalization
            thisLR = net.params(p).learningRate;
            net.params(p).value = vl_taccum(...
                1 - thisLR, net.params(p).value, ...
                (thisLR/batchSize/net.params(p).fanout), parDer);
        case 'gradient'
            thisDecay = params.weightDecay * net.params(p).weightDecay;
            thisLR = params.learningRate * net.params(p).learningRate;
            
            if thisLR>0 || thisDecay>0
                % Normalize gradient and incorporate weight decay.
                parDer = vl_taccum(1/batchSize, parDer, ...
                    thisDecay, net.params(p).value);
                
                if isempty(params.solver)
                    % Default solver is the optimised SGD.
                    % Update momentum.
                    state.solverState{p} = vl_taccum(...
                        params.momentum, state.solverState{p}, -1, parDer);
                    
                    % Nesterov update (aka one step ahead).
                    if params.nesterovUpdate
                        delta = params.momentum * state.solverState{p} - parDer;
                    else
                        delta = state.solverState{p};
                    end
                    
                    % Update parameters.
                    net.params(p).value = vl_taccum(...
                        1, net.params(p).value, thisLR, delta);
                else
                    % call solver function to update weights
                    [net.params(p).value, state.solverState{p}] = ...
                        params.solver(net.params(p).value, state.solverState{p}, ...
                        parDer, params.solverOpts, thisLR);
                end
            end
        otherwise
            error('Unknown training method ''%s'' for parameter ''%s''.', ...
                net.params(p).trainMethod, net.params(p).name);
    end
end

% -------------------------------------------------------------------------
function stats = accumulateStats(stats_)
% -------------------------------------------------------------------------
for s = {'train', 'val', 'test'}
    s_ = char(s);
    total = 0;
    
    % initialize stats stucture with same fields and same order as
    % stats_{1}
    stats__ = stats_{1};
    names = fieldnames(stats__.(s_))';
    values = zeros(1, numel(names));
    fields = cat(1, names, num2cell(values));
    stats.(s_) = struct(fields{:});
    
    for g = 1:numel(stats_)
        stats__ = stats_{g};
        num__ = stats__.(s_).num;
        total = total + num__;
        for f = setdiff(fieldnames(stats__.(s_))', 'num', 'stable')
            f_ = char(f);
            stats.(s_).(f_) = stats.(s_).(f_) + stats__.(s_).(f_) * num__;
            if g == numel(stats_)
                stats.(s_).(f_) = stats.(s_).(f_) / total;
            end
        end
    end
    stats.(s_).num = total;
end

% -------------------------------------------------------------------------
function stats = extractStats(stats, net)
% -------------------------------------------------------------------------
sel = find(cellfun(@(x) isa(x,'dagnn.Loss'), {net.layers.block}));
for i = 1:numel(sel)
    if net.layers(sel(i)).block.ignoreAverage, continue; end
    % % stats.(net.layers(sel(i)).outputs{1}) = net.layers(sel(i)).block.average;
    for j = 1:numel(net.layers(sel(i)).outputs)
        stats.(net.layers(sel(i)).outputs{j}) = net.layers(sel(i)).block.average(j);
    end
end

% -------------------------------------------------------------------------
function saveState(fileName, net_, state) %#ok
% -------------------------------------------------------------------------
net = net_.saveobj(); %#ok
save(fileName, 'net', 'state');

% -------------------------------------------------------------------------
function saveStats(fileName, stats) %#ok
% -------------------------------------------------------------------------
if exist(fileName, 'file')
    save(fileName, 'stats', '-append');
else
    save(fileName, 'stats');
end

% -------------------------------------------------------------------------
function [net, state, stats] = loadState(fileName) %#ok
% -------------------------------------------------------------------------
% % S = load(fileName, 'net', 'state', 'stats');
% % [net, state, stats] = deal(dagnn.DagNN.loadobj(S.net), S.state, S.stats);
load(fileName, 'net', 'state', 'stats');
net = dagnn.DagNN.loadobj(net); %#ok
if isempty(whos('stats'))
    error('Epoch ''%s'' was only partially saved. Delete this file and try again.', ...
        fileName);
end

% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*.mat'));
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens');
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens);
epoch = max([epoch 0]);

% -------------------------------------------------------------------------
function switchFigure(n)
% -------------------------------------------------------------------------
if get(0, 'CurrentFigure') ~= n
    try
        set(0, 'CurrentFigure', n);
    catch
        figure(n);
    end
end

% -------------------------------------------------------------------------
function clearMex()
% -------------------------------------------------------------------------
clear vl_tmove vl_imreadjpeg;

% -------------------------------------------------------------------------
function prepareGPUs(opts, cold)
% -------------------------------------------------------------------------
numGpus = numel(opts.gpus);
if numGpus > 1
    % check parallel pool integrity as it could have timed out
    pool = gcp('nocreate');
    if ~isempty(pool) && pool.NumWorkers ~= numGpus
        delete(pool);
    end
    pool = gcp('nocreate');
    if isempty(pool)
        parpool('local', numGpus);
        cold = true;
    end
end
if numGpus >= 1 && cold
    fprintf('%s: resetting GPU\n', mfilename)
    clearMex();
    if numGpus == 1
        gpuDevice(opts.gpus)
    else
        spmd
            clearMex();
            gpuDevice(opts.gpus(labindex))
        end
    end
end

% -------------------------------------------------------------------------
function a = appendElement(a, item, idx)
% -------------------------------------------------------------------------
% adds `item` to array `a` as the idx-th element
len = numel(a);
if len == 0
    a = item;
else
    a(idx) = item;
    keys = fieldnames(item)';
    % fills empty value
    for i = len:idx-1
        for k = keys
            if isempty(a(i).(k{1})), a(i).(k{1}) = NaN; end
        end
    end
end
