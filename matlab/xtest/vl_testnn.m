function vl_testnn(varargin)
%VL_TESTNN Run MatConvNet test suite 
%
% VL_TESTNN('option', value, ...) takes the following options:
%  `cpu`:: true
%    Run the CPU tests.
%
%  `gpu`:: false
%    Run the GPU tests.
%
%  `command`:: 'nn'
%    Run only tests which name starts with the specified substring.
%    E.g. `vl_testnn('command', 'nnloss') would run only the nnloss tests.
%
%  `break`:: false
%    Stop tests in case of error.
%
%  `tapFile`:: ''
%    Output the test results to a file. If a specified file does 
%    exist it is overwritten.

% Copyright (C) 2015 Andrea Vedaldi, Karel Lenc.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.cpu = true ;
opts.gpu = false ;
opts.command = 'nn' ;
opts.break = false ;
opts.tapFile = '';
opts = vl_argparse(opts, varargin) ;

import matlab.unittest.constraints.* ;
import matlab.unittest.selectors.* ;
import matlab.unittest.plugins.TAPPlugin;
import matlab.unittest.plugins.ToFile;

% Choose which tests to run
sel = HasName(StartsWithSubstring(opts.command)) ;
if opts.cpu & ~opts.gpu
  sel = sel & HasName(ContainsSubstring('cpu')) ;
end
if opts.gpu & ~opts.cpu
  sel = sel & HasName(ContainsSubstring('gpu')) ;
end

% Run tests
root = fileparts(mfilename('fullpath')) ;
suite = matlab.unittest.TestSuite.fromFolder(fullfile(root, 'suite'), sel) ;
runner = matlab.unittest.TestRunner.withTextOutput('Verbosity',3);
if opts.break
  runner.addPlugin(matlab.unittest.plugins.StopOnFailuresPlugin) ;
end
if ~isempty(opts.tapFile)
  if exist(opts.tapFile, 'file')
    delete(opts.tapFile);
  end
  runner.addPlugin(TAPPlugin.producingOriginalFormat(ToFile(opts.tapFile)));
end
result = runner.run(suite);
display(result)
