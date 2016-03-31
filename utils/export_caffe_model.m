function export_caffe_model(net, base_output_filename, test_data)
% EXPORT_CAFFE_MODEL export a MatConvNet network to Caffe format
%   export_caffe_model(net, base_output_filename, test_data)
%   Export a MatConvNet network to Caffe format
%
%   Prerequisites:
%     - Caffe built with the new MatCaffe interface, and added to the MATLAB path
%
%   Inputs:
%     net:
%       A MatConvNet network struct.
%
%     base_output_filename:
%       Path to output file, without extension.
%
%     test_data (optional):
%       An input array or image filename, on which the models will be tested to 
%       compare their output. If not supplied, a random input will be used.
%
%   Output:
%     The following Caffe files will be created by the function:
%       [base_output_filename '.prototxt'] - Network definition file
%       [base_output_filename '.caffemodel'] - Binary Caffe model
%       [base_output_filename '_mean_image.binaryproto'] (optional) -
%         The average image that needs to be subtracted from the input
%         (if set in net.normalization.averageImage).
%
%   Author: Zohar Bar-Yehuda


[~, name] = fileparts(base_output_filename);

% convert GPU network to CPU
for idx = 1:length(net.layers)
    if isfield(net.layers{idx}, 'weights')
        net.layers{idx}.weights{1} = gather(net.layers{idx}.weights{1});
        net.layers{idx}.weights{2} = gather(net.layers{idx}.weights{2});
    end
    if isfield(net.layers{idx}, 'filters')
        net.layers{idx}.filters = gather(net.layers{idx}.filters);
    end
    if isfield(net.layers{idx}, 'filtersMomentum')
        net.layers{idx}.filtersMomentum = gather(net.layers{idx}.filtersMomentum);
    end
    if isfield(net.layers{idx}, 'biases')
        net.layers{idx}.biases = gather(net.layers{idx}.biases);
    end
    if isfield(net.layers{idx}, 'biasesMomentum')
        net.layers{idx}.biasesMomentum = gather(net.layers{idx}.biasesMomentum);
    end
end

% pre-processing

% If last layer is softmax loss, replace it with softmax
if isequal(net.layers{end}.type, 'softmaxloss') || ...
        (isequal(net.layers{end}.type, 'loss') && ...
         (~isfield(net.layers{end}, 'loss') || isequal(net.layers{end}.loss, 'softmaxlog')))
    net.layers{end}.type = 'softmax';
elseif isequal(net.layers{end}.type, 'loss')
    error('Unsupported loss: %s', net.layers{end}.loss);
end

dropout = false(size(net.layers));
for idx = 1:length(net.layers)
    % Add missing layer names
    if ~isfield(net.layers{idx}, 'name')
        net.layers{idx}.name = sprintf('layer%d', idx);
    end
    
    % copy old filters,biases notation to weights
    if isfield(net.layers{idx}, 'filters')
        net.layers{idx}.weights = {net.layers{idx}.filters net.layers{idx}.biases};
    end
    
    % mark dropout layers for deletion (not needed at test time)
    if isequal(net.layers{idx}.type, 'dropout')
        dropout(idx) = true;
    end    
end
net.layers(dropout) = []; % Remove dropout layers

if isfield(net, 'normalization') && isfield(net.normalization, 'imageSize')
    im_size = net.normalization.imageSize; 
else
    error('Missing image size. Please set net.normalization.imageSize');
end

if isfield(net, 'normalization') && isfield(net.normalization, 'averageImage')
    averageImage_size = size(net.normalization.averageImage);
    if averageImage_size(1) == 1 && averageImage_size(2) == 1
        % constant value, we'll duplicate it to im_size
        net.normalization.averageImage = repmat(net.normalization.averageImage, im_size(1), im_size(2));
    end
end

% Write prototxt file
prototxt_filename = [base_output_filename '.prototxt'];
fid = fopen(prototxt_filename, 'w');

fprintf(fid,'name: "%s"\n\n', name); % Network name

% Input dimensions
fprintf(fid, 'input: "data"\n');
fprintf(fid, 'input_dim: 1\n');
fprintf(fid, 'input_dim: %d\n', im_size(3));
fprintf(fid, 'input_dim: %d\n', im_size(1));
fprintf(fid, 'input_dim: %d\n\n', im_size(2));

dummy_data = zeros(im_size, 'single'); % Keep track of data size at each layer;

is_fully_connected = false(size(net.layers));
for idx = 1:length(net.layers)
    % write layers
    fprintf(fid,'layer {\n');
    fprintf(fid,'  name: "%s"\n', net.layers{idx}.name); % Layer name
    layer_input_size = size(dummy_data);
    switch net.layers{idx}.type
        case 'conv'
            filter_h = size(net.layers{idx}.weights{1},1);
            filter_w = size(net.layers{idx}.weights{1},2);
            if filter_h < layer_input_size(1) || ...
               filter_w < layer_input_size(2)
                % Convolution layer
                fprintf(fid, '  type: "Convolution"\n');
                write_order(fid, net.layers, idx);
                fprintf(fid, '  convolution_param {\n');
                write_kernel(fid, [filter_h, filter_w]);
                fprintf(fid, '    num_output: %d\n', size(net.layers{idx}.weights{1},4));
                write_stride(fid, net.layers{idx});
                if isfield(net.layers{idx}, 'pad') && length(net.layers{idx}.pad) == 4
                    % Make sure pad is symmetrical
                    if net.layers{idx}.pad(1) ~= net.layers{idx}.pad(2) || ...
                            net.layers{idx}.pad(3) ~= net.layers{idx}.pad(4)
                        error('Caffe only supports symmetrical padding');
                    end
                end
                write_pad(fid, net.layers{idx});
                num_groups = layer_input_size(3) / size(net.layers{idx}.weights{1},3);
                assert(mod(num_groups,1) == 0);
                if num_groups > 1
                    fprintf(fid, '    group: %d\n', num_groups);
                end
                
                fprintf(fid, '  }\n');
            elseif filter_h == layer_input_size(1) && filter_w == layer_input_size(2)
                is_fully_connected(idx) = true;
                % Fully connected layer
                fprintf(fid, '  type: "InnerProduct"\n');
                write_order(fid, net.layers, idx);
                fprintf(fid, '  inner_product_param {\n');
                fprintf(fid, '    num_output: %d\n', size(net.layers{idx}.weights{1},4));
                fprintf(fid, '  }\n');
            else
                error('Filter size (%d,%d) is larger than input size (%d,%d)', ...
                    filter_h, filter_w, layer_input_size(1), layer_input_size(2))
            end

        case 'relu'
            fprintf(fid, '  type: "ReLU"\n');
            write_order(fid, net.layers, idx);

        case 'Sigmoid'
            fprintf(fid, '  type: "ReLU"\n');
            write_order(fid, net.layers, idx);     

        case 'pool'
            fprintf(fid, '  type: "Pooling"\n');
            % Check padding compatability with caffe. See:
            % http://www.vlfeat.org/matconvnet/matconvnet-manual.pdf
            % for more details.
            if ~isfield(net.layers{idx}, 'pad')
                net.layers{idx}.pad = [0 0 0 0];
            elseif length(net.layers{idx}.pad) == 1
                net.layers{idx}.pad = repmat(net.layers{idx}.pad,1,4);
            end
            if ~isfield(net.layers{idx}, 'stride')
                net.layers{idx}.stride = [1 1];
            elseif length(net.layers{idx}.stride) == 1
                net.layers{idx}.stride = repmat(net.layers{idx}.stride,1,2);
            end
            if length(net.layers{idx}.pool) == 1
                net.layers{idx}.pool = repmat(net.layers{idx}.pool, 1, 2);
            end

            support = net.layers{idx}.pool;
            stride = net.layers{idx}.stride;
            pad = net.layers{idx}.pad;
            compatability_pad_y = ceil((layer_input_size(1)-support(1)) / stride(1)) * stride(1) ...
                + support(1) - layer_input_size(1);
            compatability_pad_x = ceil((layer_input_size(2)-support(2)) / stride(2)) * stride(2) ...
                + support(2) - layer_input_size(2);

            if pad(2) ~= pad(1) + compatability_pad_y || ...
                    pad(4) ~= pad(3) + compatability_pad_x
                % Padding is not compatible with Caffe
                error(['Padding in pooling layer net.layers{%d} is not compatible with Caffe.\n' ...
                    'For compatibility, change layer padding to:\n' ...
                    'net.layers{%d}.pad = [%d %d %d %d];'], ...
                    idx, idx, pad(1), pad(1)+compatability_pad_y, pad(3) , pad(3)+compatability_pad_x);
            end

            
            write_order(fid, net.layers, idx);
            fprintf(fid, '  pooling_param {\n');
            switch (net.layers{idx}.method)
                case 'max'
                    caffe_pool = 'MAX';
                case 'avg'
                    caffe_pool = 'AVE';
                otherwise
                    error('Unknown pooling type');
            end
            fprintf(fid, '    pool: %s\n', caffe_pool);
            write_kernel(fid, net.layers{idx}.pool);
            write_stride(fid, net.layers{idx});
            write_pad(fid, net.layers{idx});
            fprintf(fid, '  }\n');
            
        case 'normalize'
            % MATLAB param = [local_size, kappa, alpha/local_size, beta]
            fprintf(fid, '  type: "LRN"\n');
            write_order(fid, net.layers, idx);
            fprintf(fid, '  lrn_param {\n'); 
            fprintf(fid, '    local_size: %d\n', net.layers{idx}.param(1));
            fprintf(fid, '    k: %f\n', net.layers{idx}.param(2));
            fprintf(fid, '    alpha: %f\n', net.layers{idx}.param(3)*net.layers{idx}.param(1));
            fprintf(fid, '    beta: %f\n', net.layers{idx}.param(4));
            fprintf(fid, '  }\n');
            
        case 'softmax'
            fprintf(fid, '  type: "Softmax"\n');
            write_order(fid, net.layers, idx);    
            
        otherwise
            error('Unknown layer type: %s', net.layers{idx}.type);
    end
    fprintf(fid,'}\n\n');
    
    layer = struct('layers', {net.layers(idx)});
    res = vl_simplenn(layer, dummy_data);
    dummy_data = res(end).x;
end
fclose(fid);


% Write binary model file
caffe.set_mode_cpu();
caffe_net = caffe.Net(prototxt_filename,'test');
first_conv = true;
for idx = 1:length(net.layers)
    layer_type = net.layers{idx}.type;
    layer_name = net.layers{idx}.name;
    switch layer_type
        case 'conv'  
            weights = net.layers{idx}.weights{1};
            weights = permute(weights, [2 1 3 4]); % Convert from HxWxCxN to WxHxCxN per Caffe's convention
            if first_conv
                if size(weights,3) == 3
                    % We assume this is an image convolution, need to convert RGB to BGR
                    weights = weights(:,:, [3 2 1], :); % Convert from RGB to BGR channel order per Caffe's convention
                end
                first_conv = false; % Do this only for first convolution;
            end
            if is_fully_connected(idx)
                % Fully connected layer, squeeze to 2 dims
                weights = reshape(weights,[], size(weights,4));
            end
            caffe_net.layers(layer_name).params(1).set_data(weights); % set weights
            caffe_net.layers(layer_name).params(2).set_data(net.layers{idx}.weights{2}(:)); % set bias        
        case {'relu', 'normalize', 'pool', 'softmax'}
                % No weights - nothing to do                
        otherwise
            error('Unknown layer type %s', layer_type)
    end
end
model_filename = [base_output_filename '.caffemodel'];
caffe_net.save(model_filename);

% Save average image
if isfield(net, 'normalization') && isfield(net.normalization, 'averageImage')    
    mean_bgr = matlab_img_to_caffe(net.normalization.averageImage); 
    meanfile = [base_output_filename '_mean_image.binaryproto'];
    caffe.io.write_mean(mean_bgr, meanfile)
else
    meanfile = [];
end

% Test
test_time = 3; % seconds
fprintf('Testing:\n');
% Load the network we just save
caffe_net2 = caffe.Net(prototxt_filename,model_filename, 'test'); 

if ~exist('test_data', 'var')
    % test_data not supplied, use random input
    test_data = rand(im_size);
end
if ischar(test_data)
    test_data = imread(test_data);
end
if size(test_data,1) ~= im_size(1) || ...
   size(test_data,2) ~= im_size(2)
   test_data = imresize(test_data, im_size(1:2));
end

test_data = single(test_data);
img_caffe = matlab_img_to_caffe(test_data);
if isfield(net.normalization, 'averageImage');
    test_data = test_data - net.normalization.averageImage;
end

% Test on CPU
tic
iters = 0;
while toc < test_time
    resmat = vl_simplenn(net, test_data);
    iters = iters + 1;
end
fprintf('MatConvNet CPU time: %.1f msec\n', toc/iters*1000);
prob_mat = resmat(end).x;

if ~isempty(meanfile)
    mean_img_caffe = caffe.io.read_mean(meanfile);
    img_caffe = img_caffe - mean_img_caffe;
end
tic
iters = 0;
while toc < test_time
    res = caffe_net2.forward({img_caffe});
    iters = iters + 1;
end
prob_caffe = res{1};
fprintf('Caffe CPU time: %.1f msec\n', toc/iters*1000);

gpu_mode = gpuDeviceCount > 0;
if gpu_mode
    % Test on GPU
    for idx = 1:length(net.layers)
        if isfield(net.layers{idx}, 'weights')
            net.layers{idx}.weights{1} = gpuArray(net.layers{idx}.weights{1});
            net.layers{idx}.weights{2} = gpuArray(net.layers{idx}.weights{2});
        end
        if isfield(net.layers{idx}, 'filters')
            net.layers{idx}.filters = gpuArray(net.layers{idx}.filters);
        end
        if isfield(net.layers{idx}, 'filtersMomentum')
            net.layers{idx}.filtersMomentum = gpuArray(net.layers{idx}.filtersMomentum);
        end
        if isfield(net.layers{idx}, 'biases')
            net.layers{idx}.biases = gpuArray(net.layers{idx}.biases);
        end
        if isfield(net.layers{idx}, 'biasesMomentum')
            net.layers{idx}.biasesMomentum = gpuArray(net.layers{idx}.biasesMomentum);
        end
    end
    test_data = gpuArray(test_data);
    tic
    iters = 0;
    while toc < test_time
        resmat = vl_simplenn(net, test_data); 
        iters = iters + 1;
    end    
    fprintf('MatConvNet GPU time: %.2f msec\n', toc/iters*1000);
    prob_mat = resmat(end).x;
    
    caffe.set_mode_gpu();
    tic
    iters = 0;
    while toc < test_time
        res = caffe_net2.forward({img_caffe});
        iters = iters + 1;
    end
    fprintf('Caffe GPU time: %.2f msec\n', toc/iters*1000);
    prob_caffe = res{1};
end


if isa(prob_mat, 'gpuArray')
    prob_mat = gather(prob_mat);
end
fprintf('MatConvNet result: %s\n', sprintf('%f ', prob_mat));
fprintf('Caffe result: %s\n', sprintf('%f ', prob_caffe));
max_diff = max(abs(prob_mat(:) - prob_caffe(:)));
fprintf('Max MatConNet/Caffe diff: %f\n', max_diff);
assert(max_diff <= 1e-3);


function write_stride(fid, layer)
if isfield(layer, 'stride')
    if length(layer.stride) == 1
        fprintf(fid, '    stride: %d\n', layer.stride);
    elseif length(layer.stride) == 2
        fprintf(fid, '    stride_h: %d\n', layer.stride(1));
        fprintf(fid, '    stride_w: %d\n', layer.stride(2));
    end
end

function write_kernel(fid, kernel_size)
if length(kernel_size) == 1
    fprintf(fid, '    kernel_size: %d\n', kernel_size);
elseif length(kernel_size) == 2
    fprintf(fid, '    kernel_h: %d\n', kernel_size(1));
    fprintf(fid, '    kernel_w: %d\n', kernel_size(2));
end


function write_pad(fid, layer)
if isfield(layer, 'pad')
    if length(layer.pad) == 1
        fprintf(fid, '    pad: %d\n', layer.pad);
    elseif length(layer.pad) == 4
        fprintf(fid, '    pad_h: %d\n', layer.pad(1));
        fprintf(fid, '    pad_w: %d\n', layer.pad(3));
    else
        error('pad vector size must be 1 or 4')
    end
end

function write_order(fid, layers, idx)
if idx > 1
    bottom_name = layers{idx-1}.name;
else
    bottom_name = 'data';
end
if idx < length(layers)
    top_name = layers{idx}.name;
else
    top_name = 'prob';
end
fprintf(fid, '  bottom: "%s"\n', bottom_name);
fprintf(fid, '  top: "%s"\n', top_name);


function img = matlab_img_to_caffe(img)
img = single(img);
img = permute(img, [2 1 3 4]); % Convert from HxWxCxN to WxHxCxN per Caffe's convention
if size(img,3) == 3
    img = img(:,:, [3 2 1], :); % Convert from RGB to BGR channel order per Caffe's convention
end
