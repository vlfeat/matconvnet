function varargout = cnn_train_autonn(varargin)
  % Training is now handled by cnn_train_dag; this makes changes to
  % training in the master branch (i.e., for DagNN) easier to track.
  [varargout{1:nargout}] = cnn_train_dag(varargin{:}) ;
end

