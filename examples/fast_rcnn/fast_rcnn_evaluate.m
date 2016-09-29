function aps = fast_rcnn_evaluate(varargin)
%FAST_RCNN_EVALUATE  Evaluate a trained Fast-RCNN model on PASCAL VOC 2007

% Evaluate the performance of trained Fast-RCNN model on PASCAL VOC 2007
%
% Copyright (C) 2016 Hakan Bilen.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

addpath(fullfile(vl_rootnn, 'data', 'VOCdevkit', 'VOCcode'));
addpath(genpath(fullfile(vl_rootnn, 'examples', 'fast_rcnn')));

opts.dataDir   = fullfile(vl_rootnn, 'data') ;
opts.sswDir    = fullfile(vl_rootnn, 'data', 'SSW');
opts.expDir    = fullfile(vl_rootnn, 'data', 'fast-rcnn-vgg16-pascal') ;
opts.imdbPath  = fullfile(opts.expDir, 'imdb.mat');
opts.modelPath = fullfile(opts.expDir, 'net-deployed.mat') ;

opts.gpu = [] ;
opts.numFetchThreads = 1 ;
opts.nmsThresh = 0.3 ;
% heuristic: keep at most 100 detection per class per image prior to NMS
opts.max_per_image = 100 ;

opts = vl_argparse(opts, varargin) ;
display(opts);

if ~isempty(opts.gpu)
  gpuDevice(opts.gpu)
end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------
net = dagnn.DagNN.loadobj(load(opts.modelPath)) ;

net.mode = 'test' ;

if ~isempty(opts.gpu)
  net.move('gpu') ;
end

% -------------------------------------------------------------------------
%                                                   Database initialization
% -------------------------------------------------------------------------
fprintf('Loading imdb...');
if exist(opts.imdbPath,'file')==2
  imdb = load(opts.imdbPath) ;
else
  imdb = cnn_setup_data_voc07_ssw(...
    'dataDir',opts.dataDir,...
    'sswDir',opts.sswDir);
  save(opts.imdbPath,'-struct', 'imdb','-v7.3');
end

fprintf('done\n');

bopts.averageImage = net.meta.normalization.averageImage;
bopts.useGpu = numel(opts.gpu) >  0 ;
bopts.maxScale = 1000;
bopts.bgLabel = 21;
bopts.visualize = 0;
bopts.scale = 600;
bopts.interpolation = net.meta.normalization.interpolation;
bopts.numThreads = 1;

% -------------------------------------------------------------------------
%                                                                  Evaluate
% -------------------------------------------------------------------------
VOCinit;
VOCopts.testset='test';

testIdx = find(imdb.images.set == 3) ;

cls_probs  = cell(1,numel(testIdx)) ;
box_deltas = cell(1,numel(testIdx)) ;
boxscores_nms = cell(numel(VOCopts.classes),numel(testIdx)) ;
ids = cell(numel(VOCopts.classes),numel(testIdx)) ;

dataVar = 'input' ;
probVarI = net.getVarIndex('probcls') ;
boxVarI = net.getVarIndex('predbbox') ;

if isnan(probVarI)
  dataVar = 'data' ;
  probVarI = net.getVarIndex('cls_prob') ;
  boxVarI = net.getVarIndex('bbox_pred') ;
end

net.vars(probVarI).precious = true ;
net.vars(boxVarI).precious = true ;

for t=1:numel(testIdx)
  if mod(t-1,50) == 0
    fprintf('Running network %d / %d\n',t,numel(testIdx));
  end
  batch = testIdx(t);
  inputs = getBatch(bopts, imdb, batch);
  inputs{1} = dataVar ;
  net.eval(inputs) ;

  cls_probs{t} = squeeze(gather(net.vars(probVarI).value)) ;
  box_deltas{t} = squeeze(gather(net.vars(boxVarI).value)) ;
end

% heuristic: keep an average of 40 detections per class per images prior
% to NMS
max_per_set = 40 * numel(testIdx);

% detection thresold for each class (this is adaptively set based on the
% max_per_set constraint)
cls_thresholds = zeros(1,numel(VOCopts.classes));
cls_probs_concat = horzcat(cls_probs{:});


for c = 1:numel(VOCopts.classes)
  q = find(strcmp(VOCopts.classes{c}, net.meta.classes.name)) ;
  so = sort(cls_probs_concat(q,:),'descend');
  cls_thresholds(q) = so(min(max_per_set,numel(so)));
  
  for t=1:numel(testIdx)
    
    if q==2 && mod(t-1,50) == 0
      fprintf('Applying NMS %d / %d\n',t,numel(testIdx));
    end
    si = find(cls_probs{t}(q,:) >= cls_thresholds(q)) ;
    pbox = imdb.boxes.pbox{testIdx(t)};

    % back-transform bounding box corrections
    delta = box_deltas{t}(4*(q-1)+1:4*q,:)';
    pred_box = bbox_transform_inv(pbox, delta);
    
    im_size = imdb.images.size(testIdx(t),[2 1]);
    pred_box = bbox_clip(round(pred_box), im_size);

    % threshold
    boxscore = [pred_box(si,:) cls_probs{t}(q,si)'];
    [~,si] = sort(boxscore(:,5),'descend');
    boxscore = boxscore(si,:);
    boxscore = boxscore(1:min(size(boxscore,1),opts.max_per_image),:);
    
    % NMS
    pick = bbox_nms(double(boxscore),opts.nmsThresh);

    boxscores_nms{c,t} = boxscore(pick,:) ;
    ids{c,t} = repmat({imdb.images.name{testIdx(t)}(1:end-4)},numel(pick),1) ;

    if 0
      figure(1) ; clf ;
      bbox_draw(imread(fullfile(imdb.imageDir,imdb.images.name{testIdx(t)})), ...
                boxscores_nms{c,t}) ;
      title(net.meta.classes.name{q}) ;
      drawnow ;
      %pause;
      %keyboard
    end
  end  
end


%% PASCAL VOC evaluation
VOCdevkitPath = fullfile(vl_rootnn,'data','VOCdevkit');
aps = zeros(numel(VOCopts.classes),1);

% fix voc folders
VOCopts.imgsetpath = fullfile(VOCdevkitPath,'VOC2007','ImageSets','Main','%s.txt');
VOCopts.annopath   = fullfile(VOCdevkitPath,'VOC2007','Annotations','%s.xml');
VOCopts.localdir   = fullfile(VOCdevkitPath,'local','VOC2007');
VOCopts.detrespath = fullfile(VOCdevkitPath, 'results', 'VOC2007', 'Main', ['%s_det_', VOCopts.testset, '_%s.txt']);

% write det results to txt files
for c=1:numel(VOCopts.classes)
  fid = fopen(sprintf(VOCopts.detrespath,'comp3',VOCopts.classes{c}),'w');
  for i=1:numel(testIdx)
    if isempty(boxscores_nms{c,i}), continue; end
    dets = boxscores_nms{c,i};
    for j=1:size(dets,1)
      fprintf(fid,'%s %.6f %d %d %d %d\n', ...
        imdb.images.name{testIdx(i)}(1:end-4), ...
        dets(j,5),dets(j,1:4)) ;
    end
  end
  fclose(fid);
  [rec,prec,ap] = VOCevaldet(VOCopts,'comp3',VOCopts.classes{c},0);
  fprintf('%s ap %.1f\n',VOCopts.classes{c},100*ap);
  aps(c) = ap;
end
fprintf('mean ap %.1f\n',100*mean(aps));

% --------------------------------------------------------------------
function inputs = getBatch(opts, imdb, batch)
% --------------------------------------------------------------------
if isempty(batch)
  return;
end

images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
opts.prefetch = (nargout == 0);

[im,rois] = fast_rcnn_eval_get_batch(images, imdb, batch, opts);

rois = single(rois);
if opts.useGpu > 0
  im = gpuArray(im) ;
  rois = gpuArray(rois) ;
end

inputs = {'input', im, 'rois', rois} ;
