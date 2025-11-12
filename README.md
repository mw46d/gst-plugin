# Gstreamer Plugins to make use of Nvidia NPP and the Nvidia deepstream framework

## Background

I wanted to build a small Gstreamer plugin to be used in normal pipelines which would use basic NPP primitives to accomplish simple tasks. I play with little autonomous rovers to compete in friendly Robomagellan competitions (https://en.wikipedia.org/wiki/Robomagellan). Finding orange traffic cones in a video stream seemed like a good example project.
Creating a simple plugin turned out to be much more challenging than anticipated. Yes, there are Gstreamer plugin tutorials out there, but they mostly use the 'GTK style C' and there are examples from Nvidia which use their own memory management etc.
So, in the end, I built two plugins, one which uses mostly the standard gstreamer infrastructure, the NPP libraries and the additional cuda-samples already used in the courses, the second makes use of the Nvidia deepstream framework. Both follow essentially the same 'classical' image processing algorithm which I used in the process_image() method in https://github.com/mw46d/Entdecker/blob/master/ROS/mw/mw_video/nodes/image_converter.py . On my rover, I'm currently using an Oak-D-W camera with a ML cone detector, but the gstreamer idea is still an interesting experiment. 

## NPP Plugin

The NPP plugin uses NPP methods to accomplish the algorithm above:
* Works only with RGB streams, the Gstreamer pipeline has to make sure, the plugin gets a supported stream
* Creates and uses a couple of pre-allocated scratch buffers to work with.
* Copies the frame from the host to a pre-allocated device buffer.
* Does the blurring, color conversion, thresholding and feature detection via NPP.
* Adds bounding boxes 'manually' to the gstreamer frame. (I could not find a good way to do that via NPP.)
* And makes sure the pipeline continues on as expected.

I learned that 'optional' does not necessarily mean optional. Some methods, for which the documentation describes optional arguments, don't work unless complete/allocated parameters are supplied.

I added some time logging to the plugin and while the time spent in the actual plugin seems good enough (around 0.01s/frame), the overall gstreamer FPS is around 15 FPS. Somewhere a lot of time is still lost.

### Running the NPP plugin

The CUDA & NPP libraries need to be installed. Also the GStreamer dev packages need to be installed. I tested this on Ubuntu 24.04 with CUDA 12.8 and Gstreamer 1.24, but it should probably work with any close versions.

For Ubuntu 24.04, at least the following packages need to be installed:

`sudo apt-get install cuda-nvcc-12-8 cuda-libraries-dev-12-8 libnpp-dev-12-8 libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev`

The "run_npp.sh" will clone the cuda-samples when needed, build the plugin and try to run Gstreamer with the working pipeline for my laptop. That might have to change, depending on the environment. The plugin has currently the name 'mynppfilter'.

For my laptop setup, the working Gstreamer pipeline command is something like

`GST_PLUGIN_PATH=src gst-launch-1.0 v4l2src device=/dev/video0 ! 'image/jpeg,width=1920,height=1080,framerate=30/1' ! jpegdec ! videoconvert ! 'video/x-raw,format="RGB"' ! mynppfilter ! videoconvert ! fpsdisplaysink`

I tried to find a camera setup that supports at least 30 FPS of 1080p, and for my camera, that seems to work only for 'MJPEG'.

## Nvidia DeepStream Plugin

During my attempts to create a simple NPP-based gstreamer plugin, I learned about the DeepStream SDK from Nvidia. So I expanded my goals a bit to include that framework. It allows to create gstreamer pipelines where multiple steps can exchange the stream data directly in GPU/device memory. There is also a nice way to attach metadata to the 'frames', so that the different steps can contribute their results to the final outcome. 

My plugin is based on the sample plugin, so I left most of the code close to how it was. But I found a problem with the sample, it did not actually populate the cvgpumat for the processing functions. ( https://forums.developer.nvidia.com/t/ds-8-0-gst-dsexample-cuda-get-converted-mat-does-not-populate-cvgpumat/351225 )

DS 8.0 still depends on CUDA 12.8 and very specific versions of all the other Nvidia libraries. My laptop was originally already at CUDA 13.0, so I needed a lot of downgrades and 'Debian hold's to make sure, I kept the required versions.

To make full use of the CUDA advantages, I compiled OpenCV 4.12 with CUDA support for this experiment. The standard Ubuntu packages for OpenCV do not include the CUDA support! Having two versions of OpenCV requires careful separation of the library search paths at build/compile time as well as at the actual execution time!

The DeepStream Plugin
* Expects a RGB[A] stream in device memory, so any required conversions have to be done beforehand.
* Creates a cv::cuda::GpuMat from the stream for OpenCV/CUDA
* Does the blurring, color conversion & thresholding via OpenCV/CUDA
* Copies the threshold-mat to a 'host' cv::Mat
* Does the feature extraction on the CPU (I did not find an equivalent function in OpenCV/CUDA)
* Adds the interesting bounding boxes to the meta data
* Lets the pipeline continue on (The *nvdsosd* step actually draws the bounding boxes)

I added similar time logging to this plugin as well and it's faster than the NPP version. The actual frame processing takes around 0.004s/frame and the complete gstreamer pipeline with all the required DS steps can keep up with 30 FPS.

### Running the DeepStream Plugin

Running DeepStream code is much more version specific! Make sure you follow the installation instructions completely! Newer versions don't always work and the dependencies are *not* set up completely! Just because *apt-get install deepstream-8.0* is happy, does not mean you have the working version!!

Gstreamer and the gstreamer dev packages also need to be installed.

For Ubuntu 24.04, at least the following packages need to be installed:

`sudo apt-get install cuda-nvcc-12-8 cuda-libraries-dev-12-8 libnpp-dev-12-8 deepstream-8.0 libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev`

To make use of the OpenCV/CUDA code, the OpenCV sources need to be compiled with CUDA! I used 4.12 here with the following *cmake* command:

`cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D WITH_CUBLAS=1 -D WITH_CUDA=ON -D WITH_CUDNN=ON -D OPENCV_DNN_CUDA=ON -D CUDA_ARCH_BIN=8.9 -D WITH_V4L=ON -D WITH_QT=OFF -D WITH_OPENGL=ON -D WITH_GSTREAMER=ON -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_PC_FILE_NAME=opencv412.pc -D OPENCV_ENABLE_NONFREE=ON -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.12.0/modules -D INSTALL_PYTHON_EXAMPLES=OFF -D INSTALL_C_EXAMPLES=OFF -D BUILD_EXAMPLES=OFF ..`

The "run_ds.sh" will build the plugin and try to run Gstreamer with the working pipeline for my laptop. That might have to change, depending on the environment. The plugin has currently the name 'mwdsexample'.

For my laptop setup, the working Gstreamer pipeline command is something like

`GST_PLUGIN_PATH=deepstream_src gst-launch-1.0 v4l2src device=/dev/video0 ! image/jpeg,width=1920,height=1080,framerate=30/1 ! nvv4l2decoder ! m.sink_0 nvstreammux name=m batch-size=1 width=1920 height=1080 live-source=1 ! mwdsexample processing-width=1920 processing-height=1080 ! nvvideoconvert ! nvdsosd ! nveglglessink`


