function model2dot(modelPath, outPath, varargin)
%MODEL2DOT Convert a model to Graphviz dot
%  MODEL2DOT(MODEL_PATH, OUT_PATH) Generate a graphviz dot file OUT_PATH 
%  of MatConvNet model MODEL_PATH.
%
%  By default, the scripts attempts to guess the input sizes based on the
%  network normalization options and the parameter `batchSize`. However if
%  network has multiple inputs, the parameter `inputs` should be specified,
%  without that the output dot graph does not contain the variable sizes.
%
%  MODEL2DOT(..., 'Option', value) takes the following options:
%
%  `BatchSize`:: 256
%    Default batch size in case the input size guessed from net normalization.
%
%  `inputs`:: []
%    When specified, passed to `dagnn.DagNN.print` as inputs.

% Copyright (C) 2015 Karel Lenc.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'matlab', 'vl_setupnn.m'));

opts.batchSize = 256;
opts.inputs = [];
opts = vl_argparse(opts, varargin);

if ~exist(modelPath, 'file')
  error('Model %s does not exist.', modelPath);
end
fprintf('Loading %s.\n', modelPath);
obj = load(modelPath);

if isstruct(obj.layers) % DAGnn format
  net = dagnn.DagNN.loadobj(obj);
elseif iscell(obj.layers)
  net = dagnn.DagNN.fromSimpleNN(obj);
else
  error('Invalid model.');
end

inputs = opts.inputs;
inputNames = net.getInputs();
if isempty(inputs) && numel(inputNames) == 1 ...
  && isfield(obj, 'meta') && isfield(obj.meta, 'normalization') ...
  && isfield(obj.meta.normalization, 'imageSize')
  inputSize = [obj.meta.normalization.imageSize(1:3), opts.batchSize];
  fprintf('Input %s guessed to be: %s.\n', inputNames{1}, mat2str(inputSize));
  inputs = {inputNames{1}, inputSize};
end

if isempty(inputs)
  warning('Input sizes not specified.');
  dot_c = net.print('format', 'dot');
else
  dot_c = net.print(inputs, 'format', 'dot');
end

out_f = fopen(outPath, 'w');
if out_f == -1, error('Unable to open %s.', outPath); end;
fprintf(out_f, dot_c);
fclose(out_f);
fprintf('Model %s exported to %s.\n', modelPath, outPath);
