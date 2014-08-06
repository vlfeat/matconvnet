#! /bin/bash
# brief: Import various Caffe models available on the Net
# author: Andrea Vedaldi

CAFFE_URL=http://dl.caffe.berkeleyvision.org/
CAFFE_GIT=https://github.com/BVLC/caffe/raw
VGG_URL=http://www.robots.ox.ac.uk/~vgg/software/deep_eval/releases/
VGG_DEEPEVAL=deepeval-encoder-1.0.1
VGG_DEEPEVAL_MODELS=models-1.0.1

# Obtain the path of this script
pushd `dirname $0` > /dev/null
SCRIPTPATH=`pwd`
popd > /dev/null

mkdir -p data/{tmp/vgg,tmp/caffe,models}

if true
then
    (
        cd data/tmp/vgg
        ln -sf $SCRIPTPATH/proto/vgg_synset_words.txt synset_words.txt
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
        wget -c -nc $CAFFE_GIT/5d0958c173ac4d4632ea4146c538a35585a3ddc4/examples/imagenet/alexnet_deploy.prototxt
        wget -c -nc $CAFFE_GIT/8198585b4a670ee2d261d436ebecbb63688da617/examples/imagenet/imagenet_deploy.prototxt
        tar xzvf caffe_ilsvrc12.tar.gz
    )
fi

if true
then
    base=data/tmp/vgg/$VGG_DEEPEVAL/models
    in=(CNN_F CNN_M CNN_S CNN_M_128 CNN_M_1024 CNN_M_2048)
    out=(f m s m-128 m-1024 m-2048)
    synset=(caffe vgg vgg vgg vgg vgg)

    for ((i=0;i<${#in[@]};++i)); do
        out=data/models/imagenet-vgg-"${out[i]}".mat
        if test ! -e $out ; then
            python utils/import-caffe.py \
                --caffe-variant=vgg-caffe \
                --average-image=$base/mean.mat \
                --synsets=data/tmp/"${synset[i]}"/synset_words.txt \
                $base/"${in[i]}"/param.prototxt \
                $base/"${in[i]}"/model \
                $out
        else
            echo $out exists
        fi
    done
fi

if true
then
    base=data/tmp/caffe

    out=data/models/imagenet-caffe-alex.mat
    test ! -e $out && \
    python utils/import-caffe.py \
        --caffe-variant=caffe \
        --average-image=$base/imagenet_mean.binaryproto \
        --synsets=$base/synset_words.txt \
        $base/alexnet_deploy.prototxt \
        $base/caffe_alexnet_model \
        $out

    out=data/models/imagenet-caffe-ref.mat
    test ! -e $out && \
    python utils/import-caffe.py \
        --caffe-variant=caffe \
        --average-image=$base/imagenet_mean.binaryproto \
        --synsets=$base/synset_words.txt \
        $base/imagenet_deploy.prototxt \
        $base/caffe_reference_imagenet_model \
        $out
fi
