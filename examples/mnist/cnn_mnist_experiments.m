%% Experiment with the cnn_mnist_fc_bnorm

[net_bn, info_bn] = cnn_mnist(...
  'expDir', 'data/mnist-bnorm', 'batchNormalization', true);

[net_fc, info_fc] = cnn_mnist(...
  'expDir', 'data/mnist-baseline', 'batchNormalization', false);

figure(1) ; clf ;
subplot(1,2,1) ;
fc_obj = arrayfun(@(i) info_fc.val(i).objective, (1:20)'); semilogy(fc_obj', 'o-') ; hold all ;
bn_obj = arrayfun(@(i) info_bn.val(i).objective, (1:20)'); semilogy(bn_obj', '+--') ;
xlabel('Training samples [x 10^3]'); ylabel('energy') ;
grid on ;
h=legend('BSLN', 'BNORM') ;
set(h,'color','none');
title('objective') ;

subplot(1,2,2) ;
fc_top1err = arrayfun(@(i) info_fc.val(i).top1err, (1:20)'); plot(fc_top1err', 'o-') ; hold all ;
bn_top1err = arrayfun(@(i) info_bn.val(i).top1err, (1:20)'); plot(bn_top1err', '*-') ;
fc_top5err = arrayfun(@(i) info_fc.val(i).top5err, (1:20)'); plot(fc_top5err', '+-') ;
bn_top5err = arrayfun(@(i) info_bn.val(i).top5err, (1:20)'); plot(bn_top5err', 'x-') ;
h=legend('BSLN-val','BSLN-val-5','BNORM-val','BNORM-val-5') ;
grid on ;
xlabel('Training samples [x 10^3]'); ylabel('error') ;
set(h,'color','none') ;
title('top1err') ;

drawnow ;
