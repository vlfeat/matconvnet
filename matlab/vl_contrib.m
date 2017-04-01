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
opts.method = 'zip';
opts.contribUrl = 'github.com/lenck/matconvnet-contrib-test/';
opts.contribDir = fullfile(vl_rootnn(), 'contrib');
opts.method = 'zip';
opts.force = false;
opts = vl_argparse(opts, varargin);
% Support for the old-time calls
if isstr(opts.force), opts.force = str2num(opts.force); end;

if nargin < 1, action = 'avail'; end;
if nargin < 2, module = ''; end;

%TODO manage the readme as a separate repo - cache

contrib_repo = repo_factory(opts.contribUrl);
contribs_raw = webread(contrib_repo.readme_url);
contribs = parse_contributions(contribs_raw);

% Setup a module structure
if nargin > 1
  [module_found, module_id] = ismember(module, {contribs.name});
  if ~module_found
    error('Unknown module %s.');
  end
  module = contribs(module_id);
  module.path = fullfile(opts.contribDir, module.name);
  module.repo = repo_factory(module.url);
  module.setup_path = fullfile(module.path, ['setup_', module.name, '.m']);
  module.sha_path = fullfile(module.path, '.sha');
  module.present = exist(module.sha_path, 'file');
end

% Perform the command
switch lower(action)
  case 'avail'
    % TODO print with links, if available
    fprintf('\nAvailable modules:\n\n');
    arrayfun(@(s) fprintf('\t%s\t\t%s\n', s.name, s.desc), contribs);
    fprintf('\n');
  case 'get'
    switch opts.method
      case 'zip'
        get_module_zip(module, opts);
      otherwise
        error('Unknown method %s.\n');
    end
  case 'setup'
    if exist(module.setup_path, 'file')
      run(module.setup_path);
    end
  otherwise
    error('Unknown action %s.', action);
end

end

%% Parse contribution files
function contribs = parse_contributions(text)
pattern = '\*[\s]*\[(?<name>\w+)\][\s]*\((?<url>.*?)\)[\s]*(?<desc>.*?)\n';
contribs = regexp(text, pattern, 'lineanchors', 'names');
end


%% Basic ZIP Downloader
function get_module_zip(module, opts)
  function download(module, sha, opts)
    if module.present && ~opts.force
      question = sprintf('Module `%s` seems to exist. Overwrite? Y/N [N]: ', ...
        module.name);
      answer = input(question, 's');
      if isempty(answer) || ismember(lower(answer), {'n', 'no'})
        return;
      end
    end
    % TODO remove from PATH if exist
    fprintf('Donwloading %s... ', module.name);
    tmp_p = tempname();
    mkdir(tmp_p);
    unzip(module.repo.tarball_url, tmp_p);
    rmdir(module.path, 's');
    mkdir(module.path);
    movefile(fullfile(tmp_p, [module.name, '-master'], '*'), module.path, 'f');
    rmdir(tmp_p, 's');
    fd = fopen(module.sha_path, 'w');
    fprintf(fd, sha);
    fclose(fd);
    fprintf('Done\n');
  end
sha = module.repo.get_master_sha();
if ~module.present || opts.force
  download(module, sha, opts);
end
sha_fd = fopen(module.sha_path, 'r');
sha_str = fgetl(sha_fd); fclose(sha_fd);
if ~strcmp(sha, sha_str)
  download(module, sha, opts);
else
  fprintf('Module `%s` is up to date.\n', module.name);
end
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
obj.readme_url = ...
  sprintf('https://raw.githubusercontent.com/%s/%s/master/README.md', ...
  match.user, match.repo);
obj.tarball_url = ...
  sprintf('https://github.com/%s/%s/archive/master.zip', ...
  match.user, match.repo);
obj.get_master_sha = @() read_sha(match);
end

% TODO implemented BitBucket wrapper