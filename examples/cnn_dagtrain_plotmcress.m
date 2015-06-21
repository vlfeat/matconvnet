function  cnn_dagtrain_plotmcress( info, opts, varargin)
% CNN_DAGTRAIN_PLOTMCRES Plot the error rates and objectives of a
% multiclass training
opts.logobjplot = true;
opts = vl_argparse(opts, varargin);

epoch = size(info.train.objective, 2);
modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;

figure(1) ; clf ;
if isfield(info.train, 'error')
subplot(1,2,1) ;
end
if opts.logobjplot
  semilogy(1:epoch, info.train.objective, 'k') ; hold on ;
  semilogy(1:epoch, info.val.objective, 'b') ;
else
  plot(1:epoch, info.train.objective, 'k') ; hold on ;
  plot(1:epoch, info.val.objective, 'b') ;
end
xlabel('training epoch') ; ylabel('energy') ;
grid on ;
h=legend('train', 'val') ;
set(h,'color','none');
title('objective') ;
if isfield(info.train, 'error')
subplot(1,2,2) ;
plot(1:epoch, info.train.error(1,:), 'k') ; hold on ;
plot(1:epoch, info.train.error(2,:), 'k--') ;
plot(1:epoch, info.val.error(1,:), 'b') ;
plot(1:epoch, info.val.error(2,:), 'b--') ;
h=legend('train','train-5','val','val-5') ;
grid on ;
xlabel('training epoch') ; ylabel('error') ;
set(h,'color','none') ;
title('error') ;
drawnow ;
end
print(1, modelFigPath, '-dpdf') ;

end

