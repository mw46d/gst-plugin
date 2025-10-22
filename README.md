# Small Gstreamer Plugin to make use of NVidia NPP

## Background

I wanted to build a small Gstreamer plugin to be used in normal pipelines which would use basic NPP primitives to accomplish simple tasks.
That turned out to be much more challenging than anticipated. Yes, there are Gstreamer plugin tutorials out there, but they mostly use the 'GTK style C' and there are examples from Nvidia which use their own memory management etc. Especially the Nvidia examples pull in a lot of additional code and I wanted to limit myself just to the additional cuda-samples already used before.

## Plugin

The current plugin does not do much besides a simple proof of concept that:
* Works only with RGB streams, the Gstreamer pipeline has to make sure, the plugin gets a supported stream
* Copies the frame from the host to a pre-allocated device buffer
* Adds semi-random values to all color pixels (easy to spot)
* Copies the modified frame back to the host memory
* And makes sure the pipeline continues on as expected.

Now adding other steps to the plugin would be easy. But it's probably better to switch to the official Nvidia deepstream setup. My plugin works, but it's far from as fast as it could be. I also found, that the frame rate varies a lot probably based on the (laptop) core, the host side is running on?! It switches between almost 30 FPS (the max coming in) and around 15 FPS (?). More to play with.

## Running

The CUDA & NPP libraries need to be installed. Also the GStreamer dev packages need to be installed. I tested this on Ubuntu 24.04 with CUDA 13.0 and Gstreamer 1.24, but it should probably work with any close versions.

For Ubuntu 24.04, at least the following packages need to be installed:

`sudo apt-get install cuda-nvcc-13-0 cuda-libraries-dev-13-0 libnpp-dev-13-0 libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev`

The "run.sh" will clone the cuda-samples when needed, build the plugin and try to run Gstreamer with the working pipeline for my laptop. That might have to change, depending on the environment. The plugin has currently the name 'mynppfilter'.

For my laptop setup, the working Gstreamer pipeline command is something like

`GST_PLUGIN_PATH=src gst-launch-1.0 -vvv v4l2src device=/dev/video0 '!' image/jpeg,width=1920,height=1080,framerate=30/1 '!' jpegdec '!' videoconvert '!' 'video/x-raw,format="RGB"' '!' mynppfilter '!' videoconvert '!' fpsdisplaysink`

I tried to find a setup, which supports at least 30 FPS, and for my camera, that seems to work only for 'MJPEG'.


