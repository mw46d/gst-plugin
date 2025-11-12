#!/bin/bash

set -x

(cd deepstream_src && make clean build) || exit 1

GST_PLUGIN_PATH=deepstream_src gst-launch-1.0 v4l2src device=/dev/video0 ! image/jpeg,width=1920,height=1080,framerate=30/1 ! nvv4l2decoder ! m.sink_0 nvstreammux name=m batch-size=1 width=1920 height=1080 live-source=1 ! mwdsexample processing-width=1920 processing-height=1080 ! nvvideoconvert ! nvdsosd ! nveglglessink
