function [ outfile ] = vl_dagnn_visualise( net, res, outfile, outfmt )

% TODO avoid passing res as argument, would it be possible to compute
% automatically (needs to write something which would guess the number of
% inputs).
% TODO guess the output format.

tmpfile = tempname;
[~,netname] = fileparts(outfile);
tf = fopen(tmpfile, 'w');

out = @(varargin) fprintf(tf, varargin{:});
%out = @(varargin) fprintf(varargin{:});

inputs = struct('name', {'data', 'label'});

[arcs, bufferNames] = vl_dagnn_getarcs(net, inputs);

res_names = {res.name};
res_sizes = arrayfun(@(r) [size(r.x, 1), size(r.x, 2), size(r.x, 3)], ...
  res, 'UniformOutput', false);

out('digraph %s {\n', netname);
out('\tnode [shape = Mrecord]\n');
out('\tnorm_ex [label = "TYPE | NAME"]\n');
out('\tconv_ex [label = "CONV | NAME | {#FILTS | FILT SZ | STRIDE}", style = "filled,bold", fillcolor = "beige"]\n');
out('\tpool_ex [label = "POOL | NAME | {METHOD | POOL SZ | STRIDE}", style = "filled", fillcolor = "aliceblue"]\n');
out('\tconv_ex -> pool_ex [label="CONV OUT SZ"]\n');
out('\tpool_ex -> norm_ex [label="POOL OUT SZ"]\n');


for li = 1:numel(net.layers)
  l = net.layers{li};
  if strcmp(l.type, 'conv')
    fsz = size(l.filters);
    out('\t%s [label = "%s | %s | {#%d | %dx%d | %d, %d }", style = "filled,bold", fillcolor = "beige"]\n', ...
      l.name,  upper(l.type), l.name, fsz(4), fsz(1), fsz(2), l.stride(1), l.stride(2));
  elseif strcmp(l.type, 'pool')
    out('\t%s [label = "%s | %s | {%s | %dx%d | %d, %d}", style = "filled", fillcolor = "aliceblue"]\n', ...
      l.name, upper(l.type), l.name, l.method, l.pool(1), l.pool(2), l.stride(1), l.stride(2));
    else
      out('\t%s [label = "%s | %s"]\n', ...
      l.name, upper(l.type), l.name);
  end
end

for ai = 1:size(arcs, 2)
  s_n = bufferNames{arcs(2, ai)};
  e_n = bufferNames{arcs(3, ai)};
  [~, res_idx] = ismember(s_n, res_names);
  res_sz = res_sizes{res_idx};
  out('\t%s -> %s [label="%dx%dx%d"]\n', s_n, e_n, ...
    res_sz(1), res_sz(2), res_sz(3));
end
out('}\n');

fclose(tf);
fprintf('Running dot... ');
system(sprintf('dot -T%s %s -o %s', outfmt, tmpfile, outfile));
delete(tmpfile);
fprintf('Network visualised in %s\n', outfile);
end

