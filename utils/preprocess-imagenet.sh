#!/bin/bash
# file: preprocess-imagenet.sh
# auhtor: Andrea Vedaldi

# Use as:
#   preprocess-imagenet.sh SRC_PATH DEST_PATH
#
# The script creates a copy of the ImageNet ILSVRC CLS-LOC challenge
# data while rescaling the images. The data is supposed to be in the
# format defined by examples/cnn_imagenet_setup_data.m
#
# Images are rescaled to a height of 256 pixels.

data=$1
ram=$2

# graphics magick (faster)
num_cpus=1
method=gm

# image magick
num_cpus=8
method=im

mkdir -p "$ram"/images ;
rsync -rv --chmod=ugo=rwX "$data"/*devkit* "$ram/"

function convert_some_im()
{
    gm=gm
    out="$1"
    shift
    for infile in "$@"
    do
        outfile="$out/$(basename $infile)"
        if test -f "$outfile"
        then
            continue ;
        fi
        convert -verbose -quality 90 -resize '256x256^' \
            "$infile" JPEG:"${outfile}.temp"
        mv "${outfile}.temp" "$outfile"
    done
}
export convert_some_im

function convert_some_gm()
{
    gm=gm
    out="$1"
    shift
    for infile in "$@"
    do
        outfile="$out/$(basename $infile)"
        if test -f "$outfile"
        then
            continue ;
        fi
        echo convert -verbose "'$infile'" -quality 90 -resize 256x256^ \
            JPEG:"${outfile}"
    done | ${gm} batch -echo on -feedback on -
}
export -f convert_some_gm


dirs=$(find $data/images/* -maxdepth 2 -type d)
for d in $dirs
do
    sub=${d#${data}/images/}
    out="$ram/images/$sub"
    echo "Converting $d -> $out"
    mkdir -p "$out"
    find "$d" -maxdepth 1 -name '*.JPEG' | \
        xargs -n 1000 --max-procs=$num_cpus \
        bash -c "convert_some_$method \"$out\" \"\$@\"" _
done
