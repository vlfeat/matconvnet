%% Experiment with the cnn_mnist_fc_bnorm

[net_bn, info_bn] = cnn_mnist(...
  'expDir', 'data/mnist-bnorm', 'useBnorm', true);

[net_fc, info_fc] = cnn_mnist(...
  'expDir', 'data/mnist-baseline', 'useBnorm', false);


%%
figure(1) ; clf ;
subplot(1,2,1) ;
semilogy(info_fc.val.objective, 'k') ; hold on ;
semilogy(info_bn.val.objective, 'b') ;
xlabel('Training samples [x10^3]'); ylabel('energy') ;
grid on ;
h=legend('BSLN', 'BNORM') ;
set(h,'color','none');
title('objective') ;
subplot(1,2,2) ;
plot(info_fc.val.error, 'k') ; hold on ;
plot(info_fc.val.topFiveError, 'k--') ;
plot(info_bn.val.error, 'b') ;
plot(info_bn.val.topFiveError, 'b--') ;
h=legend('BSLN-val','BSLN-val-5','BNORM-val','BNORM-val-5') ;
grid on ;
xlabel('Training samples [x10^3]'); ylabel('error') ;
set(h,'color','none') ;
title('error') ;
drawnow ;