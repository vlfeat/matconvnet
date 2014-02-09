setenv('MW_NVCC_PATH','/Developer/NVIDIA/CUDA-5.5/bin/nvcc');
mex -f mex_gpu_opts.sh fast_conv.cu -v

% xcrun -sdk macosx10.8 clang -O -arch x86_64 
% -Wl,-syslibroot,/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.8.sdk 
% -mmacosx-version-min=10.7 -bundle 
% -Wl,-exported_symbols_list,/Applications/MATLAB_R2013a.app/extern/lib/maci64/mexFunction.map 
% -o  "test_mex.mexmaci64"  test_mex.o  
% -L/Applications/MATLAB_R2013a.app/bin/maci64 -lmx -lmex -lmat -lstdc++

% xcrun -sdk macosx10.9 clang++ -O -arch x86_64 
% -Wl,-syslibroot,/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.9.sdk 
% -mmacosx-version-min=10.9 -bundle
% -Wl,-exported_symbols_list,/Applications/MATLAB_R2013a.app/extern/lib/maci64/mexFunction.map 
% -o  "fast_conv.mexmaci64"  fast_conv.o  
% -L/Applications/MATLAB_R2013a.app/bin/maci64 -lmx -lmex -lmat -lstdc++ -lmwgpu -lcudart
