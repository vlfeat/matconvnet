function[imdb] = createDAHImdb(opts)
% createDAHImdb creates an imdb set that consists of all images in source
% and target. The data is split into train and validation sets. The
% variable set indicates if it is train or validation

% Copyright (C) 2016-17 Hemanth Venkateswara.
% All rights reserved.

if opts.isOfficeHome
    clNames = {'Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', ...
        'Bottle', 'Bucket', 'Calculator', 'Calendar', 'Candles', 'Chair', ...
        'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp', ...
        'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', ...
        'Flowers', 'Folder', 'Fork', 'Glasses', 'Hammer', 'Helmet', ...
        'Kettle', 'Keyboard', 'Knives', 'Lamp_Shade', 'Laptop', 'Marker', ...
        'Monitor', 'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan', ...
        'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 'Printer', ...
        'Push_Pin', 'Radio', 'Refrigerator', 'Ruler', 'Scissors', ...
        'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 'Soda', 'Speaker', ...
        'Spoon', 'TV', 'Table', 'Telephone', 'ToothBrush', 'Toys', ...
        'Trash_Can', 'Webcam'};
else
    clNames = {'back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', ...
        'calculator', 'desk_chair', 'desk_lamp', 'desktop_computer', ...
        'file_cabinet', 'headphones', 'keyboard', 'laptop_computer', ...
        'letter_tray', 'mobile_phone', 'monitor', 'mouse', 'mug', ...
        'paper_notebook', 'pen', 'phone', 'printer', 'projector', ...
        'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', ...
        'stapler', 'tape_dispenser', 'trash_can'};
end
labelNames_len = length(clNames);
trainValNames = {};
actLabels = {};
labels = {};
set = {};
for ii = 1:labelNames_len
    % Source
    imsSrc = dir(fullfile(opts.srcDataDir, opts.imagesSubDir, clNames{ii}, '*.jpg'));
    numSrcFiles = length(imsSrc);
    srcValSize = floor(numSrcFiles*opts.valSizeRatio);
    srcValIds = 1:srcValSize;
    srcTrainIds= srcValSize+1 : numSrcFiles;
    % Val
    for vidSrc = srcValIds
        trainValNames{end+1} = {[fullfile(opts.srcDataDir, opts.imagesSubDir, clNames{ii}), ...
            filesep, imsSrc(vidSrc).name]};
    end
    set{end+1} = 2*ones(1, length(srcValIds));
    actLabels{end+1} = ii*ones(1, length(srcValIds));
    labels{end+1} = ii*ones(1, length(srcValIds));
    % Train
    for tidSrc = srcTrainIds
        trainValNames{end+1} = {[fullfile(opts.srcDataDir, opts.imagesSubDir, clNames{ii}), ...
            filesep, imsSrc(tidSrc).name]};
    end
    set{end+1} = ones(1, length(srcTrainIds));
    actLabels{end+1} = ii*ones(1, length(srcTrainIds));
    labels{end+1} = ii*ones(1, length(srcTrainIds));
    
    % Target
    imsTgt = dir(fullfile(opts.tgtDataDir, opts.imagesSubDir, clNames{ii}, '*.jpg'));
    numTgtFiles = length(imsTgt);
    tgtValSize = floor(numTgtFiles*opts.valSizeRatio);
    tgtValIds = 1:tgtValSize;
    tgtTrainIds = tgtValSize+1 : numTgtFiles;
    % Val
    for vidTgt = tgtValIds
        trainValNames{end+1} = {[fullfile(opts.tgtDataDir, opts.imagesSubDir, clNames{ii}), ...
            filesep, imsTgt(vidTgt).name]};
    end
    set{end+1} = 2*ones(1, length(tgtValIds));
    actLabels{end+1} = ii*ones(1, length(tgtValIds));
    labels{end+1} = zeros(1, length(tgtValIds));
    % Train
    for tidTgt = tgtTrainIds
        trainValNames{end+1} = {[fullfile(opts.tgtDataDir, opts.imagesSubDir, clNames{ii}), ...
            filesep, imsTgt(tidTgt).name]};
    end
    set{end+1} = ones(1, length(tgtTrainIds));
    actLabels{end+1} = ii*ones(1, length(tgtTrainIds));
    labels{end+1} = zeros(1, length(tgtTrainIds));
end

imdb.images.trainValNames = horzcat(trainValNames{:});
imdb.images.actLabel = single(horzcat(actLabels{:})) ;
imdb.images.label = single(horzcat(labels{:})) ;
imdb.images.set = single(horzcat(set{:})) ;
imdb.images.sets = {'train', 'val'};
imdb.classes.name = single(1:labelNames_len);
imdb.classes.description = clNames;
imdb.valSizeRatio = opts.valSizeRatio;