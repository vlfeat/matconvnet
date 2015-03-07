function [ y ] = vl_nnconcat( inputs, dim, dzdy )
%VL_NNCONCAT Concatenate multiple inputs
if nargin < 2, dim = 3; end;
if nargin < 3, dzdy = []; end;

if isempty(dzdy)
  y = cat(dim, inputs{:});
else
  numdiv = numel(inputs);
  insz = size(inputs{1});
  outsz = size(dzdy);
  assert(outsz(dim) == insz(dim)*numdiv);
  
  y = cell(1, numdiv);
  divs = 1:insz(dim):(outsz(dim) + 1);
  for di = 1:numel(divs)-1
    lims = divs(di):divs(di+1)-1;
    switch dim
      case 1
        y{di} = dzdy(lims, :, :, :);
      case 2
        y{di} = dzdy(:, lims, :, :);
      case 3
        y{di} = dzdy(:, :, lims, :);
    end
  end
end

end

