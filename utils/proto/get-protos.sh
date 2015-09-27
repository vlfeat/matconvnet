# FCN
wget -nc "https://raw.githubusercontent.com/longjon/caffe/6e3916766c6b63bff07e2cfadf210ee5e46af807/src/caffe/proto/caffe.proto" --output-document=./caffe_6e3916.proto
protoc ./caffe_6e3916.proto --python_out=./
