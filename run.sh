#!/bin/bash

set -x

[ -d cuda-samples ] || git clone https://github.com/NVIDIA/cuda-samples.git || exit 1

(cd src && make clean build) || exit 1

GST_PLUGIN_PATH=src gst-launch-1.0 -vvv v4l2src device=/dev/video0 ! 'image/jpeg,width=1920,height=1080,framerate=30/1' ! jpegdec ! videoconvert ! 'video/x-raw,format="RGB"' ! mynppfilter ! videoconvert ! fpsdisplaysink
