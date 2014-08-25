fid = fopen('./utils/proto/vgg_synset_words.txt');
tline = fgetl(fid);
synset = {};
i = 1;
while ischar(tline)
    % disp(tline)
    synset(i) = {tline};
    tline = fgetl(fid);
    i = i + 1;
end
fclose(fid);

save('synset.mat', 'synset');
