# Small Gstreamer Plugin to make use of Nvidia NPP

## Background

I wanted to buld a small Gstreamer plugin to be used in normal pipelines with would use basic NPP primitives to accompish simple tasks.
That turned out to be much more challenging than anticipated. Yes, there are Gstreamer plugin tutorials out there, but they mostly use the 'GTK style C' and there are examples from Nvidia which use their own memory management etc. Especially the Nvidia examples pull in a lot of additional code and I wanted to limit myself just the the additional cuda-samples already used before.

## Plugin

The current plugin does not do much besides a simple proof of concept of
* Works only with RGB streams, the Gstreamer pipeline has to make sure, the plugin gets a supported stream
* Copying the frame from the host to a pre-allocated device buffer
* Add semi-random values to all color pixles (easy to spot)
* Copying the modified frame back to the host memory
* And making sure, the pipeline continues on as expected.

Now adding other steps to the plugin would be easy. But it's probably better to switch to the official Nvidia deepstream setup. My plugin works, but it's far from as fast as it could be. I also found, that the frame rate varies a lot probably based on the (laptop) core, the host side is running on?! It switches between almost 30 FPS (the max coming in) and around 15 FPS (?). More to play with.

## Running

The CUDA & NPP libraries need to be installed. Also the GStreamer dev packages need to be installed.

The "run.sh" will clone the cuda-samples when needed, build the plugin and try to run Gstreamer with the working pipeline for my laptop. That migt have to change, depending on the environment.

