function imdb = cnn_imagenet_setup_data(varargin)
% DATASET_SETPU   Initialize a benchmark dataset
opts.dataDir = 'data/imagenet12' ;
opts.lite = false ;
opts = vl_argparse(opts, varargin) ;

metaPath = fullfile(opts.dataDir, 'ILSVRC2012_devkit_t12', 'data', 'meta.mat') ;
valLabelsPath = fullfile(opts.dataDir, 'ILSVRC2012_devkit_t12', 'data', 'ILSVRC2012_validation_ground_truth.txt') ;
trainImageListPath = fullfile(opts.dataDir, 'imagesets', 'train.txt') ;
valImageListPath = fullfile(opts.dataDir, 'imagesets', 'val.txt') ;
testImageListPath = fullfile(opts.dataDir, 'imagesets', 'test.txt') ;

if ~exist(metaPath)
  error('Make sure that ImageNet is installed in %s (%s not found)', ...
    opts.dataDir, metaPath) ;
end

% load categories metadata
meta = load(metaPath) ;
cats = {meta.synsets(1:1000).WNID} ;
descrs = {meta.synsets(1:1000).words} ;

% load list of training images
names = textread(trainImageListPath, '%s')' ;
imageCats = regexp(names, '^[^/]+', 'match', 'once') ;
[~,labels] = ismember(imageCats, cats) ;

imdb.images.id = 1:numel(names) ;
imdb.images.name = names ;
imdb.images.set = ones(1, numel(names)) ;
imdb.images.label = labels ;
imdb.cats.label = 1:numel(cats) ;
imdb.cats.name = cats ;
imdb.cats.description = descrs ;

% load list of validation images
names = sort(textread(valImageListPath, '%s')') ;
labels = textread(valLabelsPath, '%d')' ;

imdb.images.id = horzcat(imdb.images.id, (1:numel(names)) + 1e7 - 1) ;
imdb.images.name = horzcat(imdb.images.name, names) ;
imdb.images.set = horzcat(imdb.images.set, 2*ones(1,numel(names))) ;
imdb.images.label = horzcat(imdb.images.label, labels) ;

% load list of test images
names = textread(testImageListPath, '%s')' ;
labels = zeros(1, numel(names)) ;

imdb.images.id = horzcat(imdb.images.id, (1:numel(names)) + 2e7 - 1) ;
imdb.images.name = horzcat(imdb.images.name, names) ;
imdb.images.set = horzcat(imdb.images.set, 3*ones(1,numel(names))) ;
imdb.images.label = horzcat(imdb.images.label, labels) ;

% fixes fields
imdb.images.name = strcat(imdb.images.name, '.JPEG') ;
imdb.imageDir = fullfile(opts.dataDir, 'images') ;

% sort categories by WNID (to be compatible with other implementations)
[imdb.cats.name,perm] = sort(imdb.cats.name) ;
imdb.cats.description = imdb.cats.description(perm) ;
relabel(perm) = 1:numel(imdb.cats.name) ;
ok = imdb.images.label >  0 ;
imdb.images.label(ok) = relabel(imdb.images.label(ok)) ;

if opts.lite
  % pick a small number of images for the first 10 classes
  % this cannot be done for test as we do not have test labels
  for i=1:10
    sel = find(imdb.images.label == i) ;
    train = sel(imdb.images.set(sel) == 1) ;
    val = sel(imdb.images.set(sel) == 2) ;
    train = train(1:10) ;
    val = val(1:3) ;
    keep{i} = [train val] ;
  end
  test = find(imdb.images.set == 3) ;
  keep = sort(cat(2, keep{:}, test(1:30))) ;
  imdb.images.id = imdb.images.id(keep) ;
  imdb.images.name = imdb.images.name(keep) ;
  imdb.images.set = imdb.images.set(keep) ;
  imdb.images.label = imdb.images.label(keep) ;
end
