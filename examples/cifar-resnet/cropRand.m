function images = cropRand(img)

%original image size
w = 32;
h = 32;
N = size(img,4);

%padded-image size
w_ = 40;
h_ = 40;

[tx,ty] = meshgrid(linspace(0,1,5)) ;
tfs = [tx(:)' ; ty(:)' ; zeros(1,numel(tx))] ;
tfs_ = tfs ;
tfs_(3,:) = 1 ;
tfs = [tfs,tfs_] ;

[~,transformations] = sort(rand(size(tfs,2), N), 1) ;
tf = tfs(1:2, transformations(1,:)) ;

dx = floor((w_ - w) * tf(2,:)) + 1 ;
dy = floor((h_ - h) * tf(1,:)) + 1 ;

%crop images points
images = single(zeros(h,w,size(img,3),size(img,4)));
for i=1:N
images(:,:,:,i) = img(dy(i):h+dy(i)-1,dx(i):w+dx(i)-1,:,i);
end

end