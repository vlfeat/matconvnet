#!/bin/bash
# file: preprocess-imagenet.sh
# auhtor: Andrea Vedaldi

# Use as:
#   preprocess-imagenet.sh ILSVRC2012_PATH DEST_PATH
#
# This will copy the parts of the ILSVRC2012 data (devkit) and convert
# images to have a height of 256 pixels.

data=$1
ram=$2
num_cpus=8

ln -sf $ram ./data/imagenet12-ram
mkdir -p $ram/{images,imagesets}
rsync -rv --chmod=ugo=rwX $data/ILSVRC2012_devkit_t12 $ram/
rsync -rv --chmod=ugo=rwX $data/imagesets/{train,val,test}.txt $ram/imagesets/

dirs="$data/images/val1 $data/images/val2 $(ls -d $data/images/n*)"
for d in $dirs
do
    sub=$(basename $d)
    out=$ram/images/$sub
    echo "Converting $d"
    mkdir -p "$out"
    find "$d" -name '*.JPEG' | \
    while read f; do
        test -f "$out/$(basename $f)" || echo "$f"
    done | \
        xargs -n 10 --max-procs=$num_cpus \
        convert -verbose -format jpeg -quality 90 -resize x256 \
        -set filename:f "$out/%f" -write '%[filename:f]'
done
