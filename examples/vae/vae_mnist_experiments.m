%% Training the variational autoencoder
run('../../matlab/vl_setupnn.m');
[net_bn, info_bn] = vae_mnist;

%% displaying a 2D plot of the digits in the latent space
generator=net_bn;
generator.removeLayer({'x2h','relu1','h2z_mean','h2z_log_std','z2z','loss'})
generator.renameVar('z','input');

n = 15;
digit_size = 28;
[grid_x,grid_y] = meshgrid(linspace(-15, 15, n));
grid_x=reshape(grid_x,1,1,1,[]);
grid_y=reshape(grid_y,1,1,1,[]);
z=single(cat(3,grid_x,grid_y))*0.01;
generator.eval({'input',z});
images=squeeze(generator.vars(end).value);
images2=col2im(images,[digit_size digit_size],[digit_size*n digit_size*n],'distinct')';
imagesc(images2);
colormap(jet)