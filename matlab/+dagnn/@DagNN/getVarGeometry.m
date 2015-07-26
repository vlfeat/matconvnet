function geom = getVarGeometry(self, inputs)
% GETVARGEOMETRY   Get the geometry of the DAGNN variables

% determine the set of input variables for the netowk
geom.inputs = {self.vars([self.vars.fanin] == 0).name} ;

% initialize inputs
for v = 1:numel(self.vars)
  vn = self.vars(v).name ;
  geom.vars(v).name = vn ;
  geom.vars(v).size = NaN(1,4) ;
  for i = 1:numel(geom.inputs)
    in = geom.inputs{i} ;
    geom.vars(v).transforms(i).name = in ;
    geom.vars(v).transforms(i).map = NaN(6) ;
    if strcmp(in,vn)
      geom.vars(v).transforms(i).map = eye(6) ;
    end
  end
end
if nargin > 1
  for i=1:2:numel(inputs)
    v = self.getVar(inputs{i}) ;
    geom.vars(v).size = inputs{i+1} ;
  end
end

% now apply all the functions
for f = 1:numel(self.layers)
  in = self.getVar(self.layers(f).inputs) ;
  out = self.getVar(self.layers(f).outputs) ;
  pas = self.getParam(self.layers(f).params) ;

  [outputSizes, transforms] = self.layers(f).block.forwardGeometry(...
    {geom.vars(in).size}, ...
    cellfun(@size, {self.params(pas).value}, 'UniformOutput', false)) ;

  % blend each input parameter in each output parameter
  for neti = 1:numel(geom.inputs)
    for funco = 1:numel(out)
      funcov = out(funco) ;
      geom.vars(funcov).size = outputSizes{funco} ;
      tfs = {} ;
      for funci = 1:numel(in)
        funciv = in(funci) ;
        tfs{funci} = geom.vars(funciv).transforms(neti).map * transforms{funci,funco} ;
      end
      geom.vars(funcov).transforms(neti).map = accumtfs(tfs) ;
    end
  end
end
end

function tf = accumtfs(tfs)
tf = NaN(6) ;
discard = [] ;
for i=1:numel(tfs), discard(i) = any(isnan(tfs{i}(:))) ; end
tfs = tfs(~discard) ;
if isempty(tfs), return ; end
tf = tfs{1} ;
bad = false ;
for i=2:numel(tfs)
  bad = bad | (tfs{i}(1,1) ~= tf(1)) | (tfs{i}(2,2) ~= tf(2,2)) ;
  tf(1:2,3) = min(tf(1:2,3), tfs{i}(1:2,3)) ;
  tf(4:5,6) = max(tf(4:5,6), tfs{i}(4:5,6)) ;
end
end
