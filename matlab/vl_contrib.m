function res = vl_contrib(command, module, varargin)
%VL_CONTRIB Contribution modules management
%   VL_CONTRIB is a tool to download, set up and compile external
%   contribution modules for MatConvNet. It downloads the list of available
%   modules from a dedicated repository and uses GIT (if present) or ZIP
%   files to install the modules. Additionally it can setup, compile and
%   test the modules.
%
%   VL_CONTRIB (with no arguments) shows the list of available modules, and
%   if possible shows hyperlinks with the available commands for each
%   module.
%
%   VL_CONTRIB('COMMAND', 'MODULE', ...), or
%   VL_CONTRIB COMMAND MODULE ... if all parameters are strings, executes
%   one of the following commands:
%
%   `VL_CONTRIB LIST`::
%      Print a list of available modules.
%
%   `VL_CONTRIB INSTALL MODULE`::
%      Install a module. Specify the `'force', true` option
%      to overwrite the existing module. Modules are installed in
%      ``<vl_rootnn()>/contrib`.
%
%   `VL_CONTRIB UPDATE MODULE`::
%      Update a module. Specify the `'force', true` option
%      to overwrite the existing module.
%
%   `VL_CONTRIB SETUP MODULE`::
%      Setup the MATLAB path for a MODULE so that it can be used.
%      This is equivalent to running
%      `<vl_rootnn()>/contrib/MODULE/setup_MODULE.m`.
%
%   `VL_CONTRIB UNLOAD MODULE`::
%      Remove the module from the MATLAB path.
%
%   `VL_CONTRIB COMPILE MODULE ...`::
%      Compile a MODULE. See the module documentation for additional
%      details and accepted arguments.
%      This is equivalent to running
%      `<vl_rootnn()>/contrib/MODULE/compile_MODULE.m`.
%
%   `VL_CONTRIB TEST MODULE ...`::
%      Test a MODULE, if a test script or a test suite dir exists.
%      Test script path is: `<vl_rootnn()>/contrib/MODULE/test_MODULE.m`
%      Test suite dir is:
%      `<vl_rootnn()>/contrib/MODULE/xtest/suite/`. See `vl_testnn` for
%      additional arguments.
%
%   `VL_CONTRIB PATH MODULE ...`::
%      Return the path of a MODULE.
%
%   ## Notes
%
%   The list of modules is hosted at `github.com/vlfeat/matconvnet-contrib`.
%
%   Modules are installed to the directory `<vl_rootnn()>/contrib` in
%   MatConvNet root directory. This directory must be writable for installation
%   to succeed.
%
%   Modules are installed using GIT, if available; otherwise the function
%   unpacks the zip distribution of the modules from its repository.
%
%   See also: VL_SETUPNN, VL_COMPILENN, VL_TESTNN.

% Copyright (C) 2017 Karel Lenc and Joao Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
assert(exist('vl_argparse', 'file')==2, 'MCN not set up, run vl_setupnn.')

opts.contribUrl = 'github.com/vlfeat/matconvnet-contrib/';
opts.contribDir = fullfile(vl_rootnn(), 'contrib');
if has_git(), opts.method = 'git'; else, opts.method = 'zip'; end
opts.force = false;
opts.showlinks = usejava('desktop');
[opts, varargin] = vl_argparse(opts, varargin);
% Support for the script-like calls
if isstr(opts.force), opts.force = str2num(opts.force); end;

if nargin < 1, command = 'list'; end;
if nargin < 2, module = ''; end;

% Persistent to reduce the reads
persistent contribs;
if isempty(contribs)
  contrib_repo = repo_factory(opts.contribUrl);
  try 
    contribs_raw = webread(contrib_repo.readme_url);
    mkdir(opts.contribDir) ;
    write_file(fullfile(opts.contribDir, 'contribs.txt'),contribs_raw) ;
  catch
    contribs_raw = read_file(fullfile(opts.contribDir, 'contribs.txt')) ;
  end
  contribs = parse_contributions(contribs_raw);
end

% Setup a module structure
if nargin > 1
  [module_found, module_id] = ismember(module, {contribs.name});
  if ~module_found, error('Unknown module ''%s''.', module); end
  module = module_init(contribs(module_id), opts);
end

switch lower(command)
  case 'list'
    if nargout > 0, res = {contribs.name}; else
      fprintf('\nAvailable modules:\n\n');
      arrayfun(@(s) module_print(s, opts), contribs);
      fprintf('\n');
    end
  case {'install', 'update'}
    module_get(module, opts);
  case 'compile'
    module_compile(module, varargin{:});
  case 'setup'
    module_setup(module, varargin{:});
  case 'unload'
    module_unload(module);
  case 'test'
    module_test(module, varargin{:});
  case 'url'
    res = module.repo.url;
  case 'path'
    res = module.path;
  otherwise
    error('Unknown command %s.', command);
end
end

% --------------------------------------------------------------------
%                                                     Module functions
% --------------------------------------------------------------------

% --------------------------------------------------------------------
function module = module_init(module, opts)
% --------------------------------------------------------------------
name2path = @(name) strrep(name, '-', '_');
module.path = fullfile(opts.contribDir, module.name);
module.present = exist(module.path, 'dir');
module.repo = repo_factory(module.url);
module.setup_path = fullfile(module.path, ...
  ['setup_', name2path(module.name), '.m']);
module.compile_path = fullfile(module.path, ...
  ['compile_', name2path(module.name), '.m']);
module.test_path = fullfile(module.path, ...
  ['test_', name2path(module.name), '.m']);
module.test_dir = fullfile(module.path, 'xtest', 'suite');
module.sha_path = fullfile(module.path, '.sha'); % For ZIP only
end

% --------------------------------------------------------------------
function module_print(module, opts)
% --------------------------------------------------------------------
module = module_init(module, opts);
if opts.showlinks
  fprintf('\t<a href="%s">%s</a>', module.url, module.name);
  fprintf('\t%s\n\t\t', module.desc);
  if exist(module.path, 'dir')
    fprintf('<a href="matlab: vl_contrib update %s">[Update]</a> ', ...
      module.name);
    if module_loaded(module)
      fprintf('<a href="matlab: vl_contrib unload %s">[Unload]</a> ', ...
        module.name);
    elseif exist(module.setup_path, 'file')
      fprintf('<a href="matlab: vl_contrib setup %s">[Setup]</a> ', ...
        module.name);
    end
    if exist(module.compile_path, 'file')
      fprintf('<a href="matlab: vl_contrib compile %s">[Compile]</a> ', ...
        module.name);
    end
    if exist(module.test_path, 'file') || exist(module.test_dir, 'dir')
      fprintf('<a href="matlab: vl_contrib test %s">[Test]</a> ', ...
        module.name);
    end
    fprintf('(%s)', get_module_type(module));
  else
    fprintf('<a href="matlab: vl_contrib install %s">[Install]</a> ', ...
      module.name);
  end
  fprintf('\n\n');
else
  fprintf('\t%s\t\t%s', module.name, module.desc);
  if exist(module.path, 'dir')
    fprintf(' (%s)', get_module_type(module));
  end
  fprintf('\n');
end
end

% --------------------------------------------------------------------
function module_get(module, opts)
% --------------------------------------------------------------------
if exist(module.path, 'dir')
  method = get_module_type(module);
  if isempty(method)
    error('Module ''%s'' does not seem to be managed by vl_contrib.', ...
      module.name);
  end;
else
  method = opts.method;
end
switch method
  case 'git'
    git_get_module(module, opts);
  case 'zip'
    zip_get_module(module, opts);
  otherwise
    error('Unknown method ''%s''.', opts.method);
end
fprintf('Module %s updated.\n', module.name);
if opts.showlinks
  pcmd = @(cmd) fprintf('<a href="matlab: vl_contrib %s %s">%s</a>', ...
    cmd, module.name, cmd);
else
  pcmd = @(cmd) fprintf('`vl_contrib %s %s`', cmd, module.name);
end
if ~module_loaded(module)
  fprintf('To set up the module, run: ');
  if exist(module.compile_path, 'file')
    pcmd('compile'); fprintf(' and ');
  end
  pcmd('setup'); fprintf('\n');
end
end

% --------------------------------------------------------------------
function module_compile(module, varargin)
% --------------------------------------------------------------------
if exist(module.compile_path, 'file')
  module_setup(module);
  [compile_dir, compile_nm, ~] = fileparts(module.compile_path);
  addpath(compile_dir);  % needed to be able to call the function
  handle = str2func(compile_nm);
  handle(varargin{:});
end
end

% --------------------------------------------------------------------
function res = module_loaded(module)
% --------------------------------------------------------------------
paths = strsplit(path(), ':');
is_modpath = cellfun(@(s) ~isempty(strfind(s, module.path)), paths);
res = any(is_modpath);
end

% --------------------------------------------------------------------
function module_setup(module, varargin)
% --------------------------------------------------------------------
if exist(module.setup_path, 'file')
  run(module.setup_path, varargin{:});
else
  % if no setup function, add the default locations to path: the root
  % and the matlab subdirectory.
  addpath(module.path);
  if exist([module.path '/matlab'], 'dir')
    addpath([module.path '/matlab']);
  end
end
end

% --------------------------------------------------------------------
function to_unload = module_unload(module)
% --------------------------------------------------------------------
paths = strsplit(path(), ':');
is_modpath = cellfun(@(s) ~isempty(strfind(s, module.path)), paths);
to_unload = any(is_modpath);
if to_unload
  modpaths = paths(is_modpath);
  rmpath(modpaths{:});
end
end

% --------------------------------------------------------------------
function module_test(module, varargin)
% --------------------------------------------------------------------
if exist(module.test_path, 'file')
  run(module.test_path, varargin{:});
elseif exist(module.test_dir, 'dir')
  vl_testnn('suiteDir', module.test_dir, varargin{:});
end
end

% --------------------------------------------------------------------
%                                                   Download functions
% --------------------------------------------------------------------

% --------------------------------------------------------------------
function res = is_git(module)
% --------------------------------------------------------------------
res = exist(fullfile(module.path, '.git'), 'dir');
end

% --------------------------------------------------------------------
function git_get_module(module, opts)
% --------------------------------------------------------------------
  function ret = git(varargin)
    % TODO solve this issue better than just disabling the SSL
    cmd = [{'GIT_SSL_NO_VERIFY=true', 'git'}, varargin];
    cmd = strjoin(cmd, ' ');
    [ret, str] = system(cmd);
    if ret ~= 0
      error('Git failed:\n%s', str);
    end
  end

if exist(module.path, 'dir')
  if ~is_git(module)
    error('Module %s in %s is not a git repo. Change method?', ...
      module.name, module.path)
  end
  git(...
    '--git-dir', fullfile(module.path, '.git'), ...
    '--work-tree', module.path, ...
    'pull');
else
  fprintf('Initialising git repository ''%s''.\n', module.path);
  git('clone', module.repo.git_url, module.path);
end
end

% --------------------------------------------------------------------
function res = is_zip(module)
% --------------------------------------------------------------------
res = exist(module.sha_path, 'file');
end

% --------------------------------------------------------------------
function res = zip_get_module(module, opts)
% --------------------------------------------------------------------
  function download(module, sha, opts)
    if module.present && ~opts.force
      question = sprintf(...
        'Module `%s` seems to exist. Reinstall? Y/N [N]: ', ...
        module.name);
      if ~verify_user(question), return; end;
    end
    unloaded = module_unload(module);
    fprintf('Donwloading ''%s''... ', module.name);
    tmp_p = tempname();
    mkdir(tmp_p);
    unzip(module.repo.tarball_url, tmp_p);
    if exist(module.path, 'dir')
      rmdir(module.path, 's');
    end
    mkdir(module.path);
    tmp_dir = fullfile(tmp_p, [module.name, '-master'], '*');
    movefile(tmp_dir, module.path, 'f');
    rmdir(tmp_p, 's');
    fd = fopen(module.sha_path, 'w');
    fprintf(fd, sha);
    fclose(fd);
    fprintf('Done\n');
    if unloaded, module_setup(module); end;
  end

sha = module.repo.get_master_sha();
if ~is_zip(module) || opts.force
  download(module, sha, opts);
  res = true;
  return;
end
sha_fd = fopen(module.sha_path, 'r');
sha_str = fgetl(sha_fd); fclose(sha_fd);
if ~strcmp(sha, sha_str)
  download(module, sha, opts);
else
  fprintf('Module `%s` is up to date.\n', module.name);
end
res = true;
end

% --------------------------------------------------------------------
%                                                  Repository wrappers
% --------------------------------------------------------------------

% --------------------------------------------------------------------
function repo = repo_factory(url)
% --------------------------------------------------------------------
repo = github(url);
if ~isempty(repo), return; end;
error('Invalid repository %s.', url);
end

% --------------------------------------------------------------------
function obj = github(url)
% --------------------------------------------------------------------
  function sha = read_sha(match)
    master_sha_url = ...
      sprintf('https://api.github.com/repos/%s/%s/git/refs/heads/master',...
      match.user, match.repo);
    data = webread(master_sha_url);
    sha = data.object.sha;
  end

pattern = 'github\.com/(?<user>[^/]*)/(?<repo>[^/]*)';
match = regexp(url, pattern, 'names');
if isempty(match), obj = []; return; end;
obj.url = url;
obj.git_url = [url, '.git'];
obj.readme_url = ...
  sprintf('https://raw.githubusercontent.com/%s/%s/master/README.md', ...
  match.user, match.repo);
obj.tarball_url = ...
  sprintf('https://github.com/%s/%s/archive/master.zip', ...
  match.user, match.repo);
obj.get_master_sha = @() read_sha(match);
end

% TODO implemented BitBucket wrapper

% --------------------------------------------------------------------
%                                                    Utility functions
% --------------------------------------------------------------------

% --------------------------------------------------------------------
function res = has_git()
% --------------------------------------------------------------------
[ret, ~] = system('which git');
res = ret == 0;
end

% --------------------------------------------------------------------
function contribs = parse_contributions(text)
% --------------------------------------------------------------------
pattern = '\*[\s]*\[(?<name>[\w-]+)\][\s]*\((?<url>.*?)\)[\s]*(?<desc>.*?)\n';
contribs = regexp(text, pattern, 'lineanchors', 'names');
end

% --------------------------------------------------------------------
function method = get_module_type(module)
% --------------------------------------------------------------------
if is_git(module), method = 'git'; return; end;
if is_zip(module), method = 'zip'; return; end;
method = '';
end

% --------------------------------------------------------------------
function data = read_file(filePath)
% --------------------------------------------------------------------
f = fopen(filePath,'r');
data = fread(f, +inf, 'uint8=>char') ;
fclose(f) ;
data = data(:)' ;
end

% --------------------------------------------------------------------
function data = write_file(filePath,data)
% --------------------------------------------------------------------
f = fopen(filePath,'w');
fwrite(f, data, 'char') ;
fclose(f) ;
end

% --------------------------------------------------------------------
function res = verify_user(question)
% --------------------------------------------------------------------
answer = input(question, 's');
res = ~(isempty(answer) || ismember(lower(answer), {'n', 'no'}));
end
