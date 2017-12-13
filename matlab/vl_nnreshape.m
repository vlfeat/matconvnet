function Y = vl_nnreshape(X, sz)
szX = [size(X,1), size(X,2), size(X,3), size(X,4)];
szY = [sz(1:3) szX(4)];
Y = reshape(X, szY);

