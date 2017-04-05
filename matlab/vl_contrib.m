function vl_contrib(action, module, varargin)
%VL_CONTRIB Contribution modules managemenr
%  VL_CONTRIB AVAIL
%    Prints a list of available modules.
%
%  VL_CONTRIB GET MODULE
%    Downloads a module MODULE.
%
%  VL_CONTRIB SETUP MODULE
%    Setups a module MODULE.
%
%   See also: VL_NNSETUP, VL_NNCOMPILE, VL_TESTNN.

% Copyright (C) 2017 Karel Lenc.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
opts.contribUrl = 'github.com/lenck/matconvnet-contrib-test/';
opts.contribDir = fullfile(vl_rootnn(), 'contrib');
if has_git(), opts.method = 'git'; else, opts.method = 'zip'; end
opts.force = false;
opts = vl_argparse(opts, varargin);
% Support for the old-time calls
if isstr(opts.force), opts.force = str2num(opts.force); end;

if nargin < 1, action = 'avail'; end;
if nargin < 2, module = ''; end;

% Persistent to reduce the reads
persistent contribs;
if isempty(contribs)
  contrib_repo = repo_factory(opts.contribUrl);
  contribs_raw = webread(contrib_repo.readme_url);
  contribs = parse_contributions(contribs_raw);
end

% Setup a module structure
if nargin > 1
  [module_found, module_id] = ismember(module, {contribs.name});
  if ~module_found, error('Unknown module %s.'); end
  module = contribs(module_id);
  module.path = fullfile(opts.contribDir, module.name);
  module.repo = repo_factory(module.url);
  module.setup_path = fullfile(module.path, ['setup_', module.name, '.m']);
  module.compile_path = fullfile(module.path, ['compile_', module.name, '.m']);
  module.sha_path = fullfile(module.path, '.sha');
end

% Perform the command
switch lower(action)
  case 'avail'
    avail_modules(contribs);
  case {'install', 'update'}
    get_module(module, opts);
  case 'compile'
    compile_module(module);
  case 'setup'
    setup_module(module);
  case 'unload'
    unload_module(module);
  otherwise
    error('Unknown action %s.', action);
end

end

function avail_modules(contribs)
fprintf('\nAvailable modules:\n\n');
arrayfun(@(s) fprintf('\t%s\t\t%s\n', s.name, s.desc), contribs);
fprintf('\n');
end

function get_module(module, opts)
switch opts.method
  case 'git'
    res = get_module_git(module, opts);
  case 'zip'
    get_module_zip(module, opts);
  otherwise
    error('Unknown method %s.', opts.method);
end
end

function setup_module(module)
if exist(module.setup_path, 'file')
  run(module.setup_path);
end
end

function compile_module(module)
if exist(module.compile_path, 'file')
  run(module.compile_path);
end
end

function to_unload = unload_module(module)
paths = strsplit(path(), ':');
is_modpath = cellfun(@(s) ~isempty(strfind(s, module.path)), paths);
to_unload = any(is_modpath);
if to_unload
  modpaths = paths(is_modpath);
  rmpath(modpaths{:});
end
end

%% Parse contribution files
function contribs = parse_contributions(text)
pattern = '\*[\s]*\[(?<name>\w+)\][\s]*\((?<url>.*?)\)[\s]*(?<desc>.*?)\n';
contribs = regexp(text, pattern, 'lineanchors', 'names');
end

function res = has_git()
[ret, ~] = system('which git');
res = ret == 0;
end

%% GIT Downloader
function res = get_module_git(module, opts)
  function ret = git(varargin)
    % TODO solve this issue better than just disabling the SSL
    cmd = {'GIT_SSL_NO_VERIFY=true', 'git'};
    cmd = [cmd, varargin];
    cmd = strjoin(cmd, ' ');
    [ret, str] = system(cmd);
    if ret ~= 0
      error('Git failed:\n%s', str);
    end
  end

ret = inf;
if exist(module.path, 'dir')
  if ~exist(fullfile(module.path, '.git'), 'dir')
    error('Module %s in %s is not a git repo. Change method?', ...
      module.name, module.path)
  end
  ret = git('--git-dir', fullfile(module.path, '.git'), 'pull');
else
  fprintf('Initialising git repository %s.\n', module.path);
  ret = git('clone', module.repo.git_url, module.path);
end
res = ret == 0;
end

%% Basic ZIP Downloader
function res = get_module_zip(module, opts)
  function download(module, sha, opts)
    if module.present && ~opts.force
      question = sprintf(...
        'Module `%s` seems to exist. Reinstall? Y/N [N]: ', ...
        module.name);
      answer = input(question, 's');
      if isempty(answer) || ismember(lower(answer), {'n', 'no'})
        return;
      end
    end
    unloaded = unload_module(module);
    fprintf('Donwloading %s... ', module.name);
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
    if unloaded, setup_module(module); end;
  end

sha = module.repo.get_master_sha();
present = exist(module.sha_path, 'file');
if ~present || opts.force
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

%% Repository wrappers
function repo = repo_factory(url)
repo = github(url);
if ~isempty(repo), return; end;
error('Invalid repository %s.', url);
end

function obj = github(url)
  function sha = read_sha(match)
    master_sha_url = ...
      sprintf('https://api.github.com/repos/%s/%s/git/refs/heads/master',...
      match.user, match.repo);
    data = webread(master_sha_url);
    sha = data.object.sha;
  end

pattern = 'github\.com/(?<user>[\w]*)/(?<repo>[\w-]*)';
match = regexp(url, pattern, 'names');
if isempty(match), obj = []; return; end;
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