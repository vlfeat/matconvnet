function convert_matconvnet_caffe(net, base_output_filename, test_img)
% CONVERT_MATCONVNET_CAFFE export a MatConvNet network to Caffe format
%   convert_matconvnet_caffe(net, base_output_filename, test_img)
%   Export a MatConvNet network to Caffe format
%
%   Prerequisites:
%     - MatConvNet
%     - Caffe built with the new MatCaffe interface
%
%   Inputs:
%     net:
%       A MatConvNet network struct.
%
%     base_output_filename:
%       Path to output file, without extension.
%
%     test_img:
%       An image filename or array, on which the models will be tested to 
%       compare their output.
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

if ismember(net.layers{end}.type, {'softmaxloss', 'weightedsoftmaxloss'})
    net.layers{end}.type = 'softmax';
end
for idx = 1:length(net.layers)
    if ~isfield(net.layers{idx}, 'name')
        net.layers{idx}.name = sprintf('layer%d', idx);
    end
    if isfield(net.layers{idx}, 'filters')
        % copy old filters,biases notation to weights
        net.layers{idx}.weights = {net.layers{idx}.filters net.layers{idx}.biases};
    end
end

if isfield(net, 'normalization') && isfield(net.normalization, 'imageSize')
    im_size = net.normalization.imageSize; 
elseif isfield(net, 'normalization') && isfield(net.normalization, 'averageImage')
    im_size = size(net.normalization.averageImage);
else
    error('Missing image size. Please set net.normalization.imageSize');
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

for idx = 1:length(net.layers)
    % write layers
    fprintf(fid,'layer {\n');
    fprintf(fid,'  name: "%s"\n', net.layers{idx}.name); % Layer name
    switch net.layers{idx}.type
        case 'conv'
            if size(net.layers{idx}.weights{1},1) > 1 || ...
                    size(net.layers{idx}.weights{1},2) > 1
                % Convolution layer                
                fprintf(fid, '  type: "Convolution"\n');
                write_order(fid, net.layers, idx);
                fprintf(fid, '  convolution_param {\n');
                write_kernel(fid, [size(net.layers{idx}.weights{1},1), size(net.layers{idx}.weights{1},2)]);
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
                fprintf(fid, '  }\n');
            else
                % Fully connected layer
                fprintf(fid, '  type: "InnerProduct"\n');
                write_order(fid, net.layers, idx);
                fprintf(fid, '  inner_product_param {\n');                
                fprintf(fid, '    num_output: %d\n', size(net.layers{idx}.weights{1},4));                
                fprintf(fid, '  }\n');
            end
            
        case 'relu'            
            fprintf(fid, '  type: "ReLU"\n');
            write_order(fid, net.layers, idx);
            
        case 'Sigmoid'
            fprintf(fid, '  type: "ReLU"\n');
            write_order(fid, net.layers, idx);     
            
        case 'pool'            
            fprintf(fid, '  type: "Pooling"\n');
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
    end    
    fprintf(fid,'}\n\n');
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
            if size(weights,1) == 1 && size(weights,2) == 1
                % Fully connected layer, squeeze to 2 dims
                weights = squeeze(weights);
            end
            caffe_net.layers(layer_name).params(1).set_data(weights); % set weights
            caffe_net.layers(layer_name).params(2).set_data(net.layers{idx}.weights{2}'); % set bias        
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
cpu_iters = 10;
gpu_iters  = 200;
fprintf('Testing:\n');
caffe_net2 = caffe.Net(prototxt_filename,model_filename, 'test');
if ischar(test_img)
    test_img = imread(test_img);
end
if size(test_img,1) ~= im_size(1) || ...
   size(test_img,2) ~= im_size(2)
   test_img = imresize(test_img, im_size(1:2));
end

test_img = single(test_img);
img_caffe = matlab_img_to_caffe(test_img);
if isfield(net.normalization, 'averageImage');
    test_img = test_img - net.normalization.averageImage;
end
tic
for i=1:cpu_iters
    resmat = vl_simplenn(net, test_img);
end
fprintf('MatConvNet CPU time: %.1f msec\n', toc/cpu_iters*1000);
prob_mat = resmat(end).x;

if ~isempty(meanfile)
    mean_img_caffe = caffe.io.read_mean(meanfile);
    img_caffe = img_caffe - mean_img_caffe;
end
tic
for i=1:cpu_iters
    res = caffe_net2.forward({img_caffe});
end
prob_caffe = res{1};
fprintf('Caffe CPU time: %.1f msec\n', toc/cpu_iters*1000);

gpu_mode = gpuDeviceCount > 0;
if gpu_mode
    for idx = 1:length(net.layers)
        if isfield(net.layers{idx}, 'weights')
            net.layers{idx}.weights{1} = gpuArray(net.layers{idx}.weights{1});
            net.layers{idx}.weights{2} = gpuArray(net.layers{idx}.weights{2});
        end
    end
    test_img = gpuArray(test_img);
    tic
    for i=1:gpu_iters
        resmat = vl_simplenn(net, test_img);        
    end    
    fprintf('MatConvNet GPU time: %.2f msec\n', toc/gpu_iters*1000);
    prob_mat = resmat(end).x;
    
    caffe.set_mode_gpu();
    tic
    for i=1:gpu_iters
        res = caffe_net2.forward({img_caffe});
    end
    fprintf('Caffe GPU time: %.2f msec\n', toc/gpu_iters*1000);
    prob_caffe = res{1};
end


if isa(prob_mat, 'gpuArray')
    prob_mat = gather(prob_mat);
end
fprintf('MatConvNet result: %s\n', sprintf('%f ', prob_mat));
fprintf('Caffe result: %s\n', sprintf('%f ', prob_caffe));
max_diff = max(abs(prob_mat(:) - prob_caffe(:)));
fprintf('Max MatConNet/Caffe diff: %f\n', max_diff);
assert(max_diff <= 1e-5);


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
