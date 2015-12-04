function vl_testnn(varargin)
%VL_TESTNN Run MatConvNet test suite
%   VL_TESTNN('cpu', true)
%   VL_TESTNN('gpu', true)
%   VL_TESTNN('command', 'nnloss')

opts.cpu = true ;
opts.gpu = false ;
opts.command = 'nn' ;
opts.break = false ;
opts.tapFile = [];
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
  runner.addPlugin(TAPPlugin.producingOriginalFormat(ToFile(opts.tapFile)));
end
result = runner.run(suite);
display(result)
