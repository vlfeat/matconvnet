#! /bin/bash
# brief: Import FCN models from Caffe Model Zoo
# author: Karel Lenc and Andrea Vedaldi

# Models are written to <MATCONVNET>/data/models
# You can delete <MATCONVNET>/data/tmp after conversion

# TODO apply patch to prototxt which will resize the outputs of cls layers from 205 -> 1000 (maybe sed?)

FCN32S_PROTO_URL=https://gist.github.com/longjon/ac410cad48a088710872/raw/fe76e342641ddb0defad95f6dc670ccc99c35a1f/fcn-32s-pascal-deploy.prototxt
FCN16S_PROTO_URL=https://gist.githubusercontent.com/longjon/d24098e083bec05e456e/raw/dd455b2978b2943a51c37ec047a0f46121d18b56/fcn-16s-pascal-deploy.prototxt
FCN8S_PROTO_URL=https://gist.githubusercontent.com/longjon/1bf3aa1e0b8e788d7e1d/raw/2711bb261ee4404faf2ddf5b9d0d2385ff3bcc3e/fcn-8s-pascal-deploy.prototxt
FCNALEX_PROTO_URL=https://gist.githubusercontent.com/shelhamer/3f2c75f3c8c71357f24c/raw/ccd0d97662e03b83e62f26bf9d870209f20f3efc/train_val.prototxt

FCN32S_MODEL_URL=http://dl.caffe.berkeleyvision.org/fcn-32s-pascal.caffemodel
FCN16S_MODEL_URL=http://dl.caffe.berkeleyvision.org/fcn-16s-pascal.caffemodel
FCN8S_MODEL_URL=http://dl.caffe.berkeleyvision.org/fcn-8s-pascal.caffemodel
FCNALEX_MODEL_URL=http://dl.caffe.berkeleyvision.org/fcn-alexnet-pascal.caffemodel

FCN_AVERAGE_COLOR="(122.67891434, 116.66876762, 104.00698793)"

FCN_CLASSES="('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')"

# Obtain the path of this script
pushd `dirname $0` > /dev/null
SCRIPTPATH=`pwd`
popd > /dev/null

converter="python $SCRIPTPATH/import-caffe-dag2.py"
data="$SCRIPTPATH/../data"

mkdir -p "$data"/tmp/caffe

# --------------------------------------------------------------------
# GoogLeNet
# --------------------------------------------------------------------

if true
then
    (
        cd "$data/tmp/caffe"
        wget -c -nc $FCN32S_MODEL_URL
        wget -c -nc $FCN32S_PROTO_URL
        wget -c -nc $FCN16S_MODEL_URL
        wget -c -nc $FCN16S_PROTO_URL
        wget -c -nc $FCN8S_MODEL_URL
        wget -c -nc $FCN8S_PROTO_URL
#        wget -c -nc $FCNALEX_MODEL_URL
#        wget -c -nc $FCNALEX_PROTO_URL
    )
fi

if true
then
    ins=(fcn-32s-pascal fcn-16s-pascal fcn-8s-pascal)
    outs=(pascal-fcn-32s-dag pascal-fcn-16s-dag pascal-fcn-8s-dag)

    #ins=(fcn-8s-pascal)
    #outs=(pascal-fcn-8s-dag)

    for ((i=0;i<${#ins[@]};++i)); do
        in="$data/tmp/caffe/${ins[i]}"
        out="$data/models/${outs[i]}.mat"
        if test ! -e "$out" ; then
            $converter \
                --caffe-variant=caffe_6e3916 \
                --preproc=fcn \
		--remove-dropout \
                --remove-loss \
                --average-value="${FCN_AVERAGE_COLOR}" \
                --class-names="${FCN_CLASSES}" \
                --caffe-data="$in".caffemodel \
                "$in"-deploy.prototxt \
                "$out"

        else
            echo "$out exists"
        fi
    done
fi
