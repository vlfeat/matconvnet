function [ imdb ] = cnn_imagenet_synchro_labels( imdb, net )
%CNN_IMAGENET_SYNCHRO_LABELS

[~,imdb.cats.label] = ismember(imdb.cats.name, net.wnid);
assert(all(imdb.cats.label ~= 0));
end

