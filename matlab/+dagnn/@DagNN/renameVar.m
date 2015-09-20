function renameVar(obj, oldName, newName)
%RENAMEVAR Rename a variable
%   RENAMEVAR(OLDNAME, NEWNAME) changes the name of the variable
%   OLDNAME into NEWNAME. NEWNAME should not be the name of an
%   existing variable.

if any(strcmp(newName, {obj.vars.name}))
  error('%s is the name of an existing variable', newName) ;
end

v = obj.getVarIndex(oldName) ;
if isnan(v)
  error('%s is not an existing variable', oldName) ;
end

for l = 1:numel(obj.layers)
  for f = {'inputs', 'outputs'}
     f = char(f) ;
     sel = find(strcmp(oldName, obj.layers(l).(f))) ;
     [obj.layers(l).(f){sel}] = deal(newName) ;
  end
end
obj.vars(v).name = newName ;

% update variable name hash otherwise rebuild() won't find this var corectly
obj.varNames.(newName) = v ;

obj.rebuild() ;
