#! /bin/bash

mkdir -p data/{tmp/vgg,tmp/caffe,models}

(
    CAFFE_URL=http://dl.caffe.berkeleyvision.org/

    VGG_URL=http://www.robots.ox.ac.uk/~vgg/software/deep_eval/releases/
    VGG_DEEPEVAL=deepeval-encoder-1.0.1
    VGG_DEEPEVAL_MODELS=models-1.0.1

    if false
    then
        (
        cd data/tmp/vgg
        wget -c -nc $VGG_URL/$VGG_DEEPEVAL.tar.gz
        tar xzvf $VGG_DEEPEVAL.tar.gz
        cd $VGG_DEEPEVAL/models
        wget -c -nc $VGG_URL/$VGG_DEEPEVAL_MODELS.tar.gz
        tar xzvf $VGG_DEEPEVAL_MODELS.tar.gz
        )
    fi

    if true
    then
        (
        cd data/tmp/caffe
        wget -c -nc $CAFFE_URL/caffe_reference_imagenet_model
        wget -c -nc $CAFFE_URL/caffe_alexnet_model
        wget -c -nc $CAFFE_URL/caffe_ilsvrc12.tar.gz
        wget -c -nc https://github.com/BVLC/caffe/raw/master/examples/imagenet/alexnet_deploy.prototxt
        wget -c -nc https://github.com/BVLC/caffe/raw/master/examples/imagenet/imagenet_deploy.prototxt
        tar xzvf caffe_ilsvrc12.tar.gz
        )
    fi

    base=data/tmp/vgg/$VGG_DEEPEVAL/models
    in=(CNN_F CNN_M CNN_S CNN_M_128 CNN_M_1024 CNN_M_2048)
    out=(f m s m-128 m-1024 m-2048)

    for ((i=0;i<${#in[@]};++i)); do
        python utils/import-caffe.py \
            --caffe-variant=vgg-caffe \
            --average-image=$base/mean.mat \
            --synsets=data/tmp/caffe/synset_words.txt \
            $base/"${in[i]}"/param.prototxt \
            $base/"${in[i]}"/model \
            data/models/imagenet-vgg-"${out[i]}".mat
    done

return

    base=data/tmp/caffe
    python utils/import-caffe.py \
        --average-image=$base/imagenet_mean.binaryproto \
        --synsets=$base/synset_words.txt \
        $base/imagenet_deploy.prototxt \
        $base/caffe_reference_imagenet_model \
        data/models/imagenet-caffe-ref.mat


   return

)
