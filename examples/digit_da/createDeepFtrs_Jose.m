function [] = createDeepFtrs_Jose()
% This function creates Deep features for the office dataset using the
% VGG_F model.

% addpath('C:\local\matconvnet-1.0-master\matlab');
% 
% dataDir = 'D:\Dataset10062016v2';
% codeDir = 'C:\local\DomainAdaption\Networks';

% Do you want Hash features (16 bit or 64 bit after fc8) or fc7 features (after fc7)
is_fc4 = false;
if is_fc4
    fc4 = '-fc4';
else
    fc4 = '';
end

netRoot = '/data/DB';
dataRoot = '/home/ASUAD/hkdv1/CodeRep/MatConvNet/da_hash/examples';

%dataset = 'Product';
%metaPath = fullfile(dataDir, 'meta.mat');
% netPath = '/data/DB/OfficeHome/imagenet-vgg-f.mat';

srcsDi = {'mnist', 'usps'};
tgtsDi = {'usps', 'mnist'};
% srcsOH = {'Art','Art','Art','Clipart'};
% tgtsOH = {'Clipart','Product','RealWorld','Art'};
% metafileOH = {'metaOffice.mat','metaOffice.mat','metaOffice.mat','metaOffice.mat'};
% srcsO = {'amazon','amazon','webcam'};
% tgtsO = {'dslr','webcam','amazon'};
% metafileO = {'metaOfficeHome.mat','metaOfficeHome.mat','metaOfficeHome.mat'};


%getting both office and office home
ts = [srcsDi]; % [srcsOH, srcsO];
tt = [tgtsDi]; % [tgtsOH, tgtsO];
% tm = [metafileOH]; % ][metafileOH,metafileO];


% bittype = {'16';'64'};
bittype = {'64'};
clNames = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};
labelNames_len = length(clNames);

for domainpairindex = 1:length(ts)
   for bnumi = 1:length(bittype)
        bnum = bittype{bnumi};
        srcDir = ts{domainpairindex};
        tgtDir = tt{domainpairindex};
        dataset = 'digit_da';
        bitDir = [bnum,'bits'];
        netDir = fullfile(netRoot, dataset, [srcDir, '_', tgtDir, '-vgg-dah'], 'supW100XavierImp');
        netFile = findLastNetFile(netDir);
        netPath = fullfile(netDir, netFile);
        fprintf('\n---------------------------------------------\n');
        fprintf('Extracting %s hash values for %s using %s\n', bnum, tgtDir, netDir);
        % Extract ftrs for the target data only
        srcDataDir = fullfile(dataRoot, dataset, srcDir);
        tgtDataDir = fullfile(dataRoot, dataset, tgtDir);
        metaPath = fullfile(dataRoot, dataset, 'meta.mat');
        saveFile = [srcDir,'-To-',tgtDir,'-HashDA',bnum,fc4,'.mat'];
        savePath = fullfile(netRoot, dataset, saveFile);
        labelNames_len = length(clNames);
        srcImages = {};
        srcLabels = {};
        tgtImages = {};
        tgtLabels = {};
        for ii = 1:labelNames_len
            ims = dir(fullfile(srcDataDir, clNames{ii}, '*.jpg'));
            srcImages{end+1} = strcat([fullfile(srcDataDir, clNames{ii}), filesep], {ims.name});
            srcLabels{end+1} = ii*ones(1, numel(ims));
            ims = dir(fullfile(tgtDataDir, clNames{ii}, '*.jpg'));
            tgtImages{end+1} = strcat([fullfile(tgtDataDir, clNames{ii}), filesep], {ims.name});
            tgtLabels{end+1} = ii*ones(1, numel(ims));
        end
        srcImages = cat(2, srcImages{:});
        srcLabels = cat(2, srcLabels{:});
        tgtImages = cat(2, tgtImages{:});
        tgtLabels = cat(2, tgtLabels{:});
        imFileNames = [srcImages, tgtImages];
        imLabels = [srcLabels, tgtLabels];
        % Indicator which is source and which is target
        srcTgtId = ones(1,length(imLabels));
        srcTgtId(length(srcLabels)+1:end) = 2;
        net_cpu = load(netPath);
        net_cpu = net_cpu.net;
        bopts.numThreads = 12;
        bopts.imageSize = net_cpu.meta.inputSize ;
        bopts.subtractAverage = net_cpu.meta.normalization.averageImage ;
        
        
        if is_fc4
            ftrLevel = 10; % features after fc4
        else
            ftrLevel = 12; % before tanh and after bnorm and lmmd4
            % These can be treated as features as well.
            % To get binary codes we can do a tanh or merely threshold them.
        end
        
        numGpus = 1;
        evalMode = 'test' ;
        cudnn = true;
        sync = false ;
        conserveMemory = true ;
        backPropDepth = +inf ;
        batchSize = 400;

        if numGpus >= 1
          gpuDevice(numGpus)
          net = vl_simplenn_move(net_cpu, 'gpu') ;
          net_cpu = [];
        else
          net = net_cpu ;
          net_cpu = [] ;
        end
        
        % Get Images in batches and get their features
        bs = batchSize;
        train = 1:numel(imFileNames);
        fts = {};
        for t = 1:bs:numel(imFileNames)
            batch = train(t:min(t+bs-1, numel(imFileNames))) ;
            ims = imFileNames(batch);
            [im] = getImageBatch(ims, bopts) ;
            if numGpus >= 1
              im = gpuArray(im) ;
            end
            % Add this so that prior_entropy_loss does not fail
            net.layers{end}.class = imLabels(batch);
            
            dzdy = [];
            res = [];
            res = vl_simplenn_dah(net, im, dzdy, res, ...
                              'mode', evalMode, ...
                              'backPropDepth', backPropDepth, ...
                              'sync', sync, ...
                              'cudnn', cudnn) ;
            fprintf('\n');
            fts{end+1} = squeeze(gather(res(ftrLevel).x)) ;
            if numGpus >= 1
              net_cpu = vl_simplenn_move(net, 'cpu') ;
            else
              net_cpu = net ;
            end
        end
        fts = cat(2, fts{:});
        fts = double(fts)';
        labels = double(imLabels)';
        save(savePath, 'fts', 'labels', 'srcTgtId');
   end
end


% -------------------------------------------------------------------------
function netfile = findLastNetFile(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;
netfile = ['net-epoch-',num2str(epoch),'.mat'];

%end
% 
% % -------------------------------------------------------------------------
% function [averageImage, rgbMean, rgbCovariance] = getImageStats(images, meta)
% % -------------------------------------------------------------------------
% numFetchThreads = meta.numFetchThreads;
% bs = meta.batchSize;
% inputSize = meta.inputSize;
% avg = {}; rgbm1 = {}; rgbm2 = {};
% train = 1:numel(images);
% for t = 1:bs:numel(images)
%     batch_time = tic ;
%     batch = train(t:min(t+bs-1, numel(images))) ;
%     fprintf('collecting image stats: batch starting with image %d ...', batch(1)) ;
%     ims = vl_imreadjpeg(images(batch),'numThreads', numFetchThreads);
%     temp = single(zeros(inputSize(1),inputSize(2),3,length(batch)));
%     for bb = 1:length(batch)
%         if isempty(ims{bb})
%             imt = imread(images{bb}) ;
%             imt = single(imt) ; % faster than im2single (and multiplies by 255)
%         else
%             imt = ims{bb} ;
%         end
%         if size(imt,3) == 1
%             imt = cat(3, imt, imt, imt) ;
%         end
%         temp(:,:,:,bb) = imresize(imt, [inputSize(1), inputSize(1)]);
%     end
%     z = reshape(permute(temp,[3 1 2 4]),3,[]) ;
%     n = size(z,2) ;
%     avg{end+1} = mean(temp, 4) ;
%     rgbm1{end+1} = sum(z,2)/n ;
%     rgbm2{end+1} = z*z'/n ;
%     batch_time = toc(batch_time) ;
%     fprintf(' %.2f s (%.1f images/s)\n', batch_time, numel(batch)/ batch_time) ;
% end
% averageImage = mean(cat(4,avg{:}),4) ;
% rgbm1 = mean(cat(2,rgbm1{:}),2) ;
% rgbm2 = mean(cat(3,rgbm2{:}),3) ;
% rgbMean = rgbm1 ;
% rgbCovariance = rgbm2 - rgbm1*rgbm1' ;
% end