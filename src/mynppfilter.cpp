/**
 * Copyright (C) 2025 Marco Walther <marco@sonic.net>
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Alternatively, the contents of this file may be used under the
 * GNU Lesser General Public License Version 2.1 (the "LGPL"), in
 * which case the following provisions apply instead of the ones
 * mentioned above:
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

/**
 * SECTION:element-mynppfilter
 *
 * FIXME:Describe mynppfilter here.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v -m fakesrc ! mynppfilter ! fakesink silent=TRUE
 * ]|
 * </refsect2>
 */

#include "mynppfilter.h"

#include <alloca.h>
#include <iostream>

#include <gst/controller/controller.h>

G_DEFINE_TYPE (GstMyNppFilter, gst_my_npp_filter, GST_TYPE_BASE_TRANSFORM)

GST_DEBUG_CATEGORY_STATIC (gst_my_npp_filter_debug);
#define GST_CAT_DEFAULT gst_my_npp_filter_debug

// Filter signals and args
enum
{
  // FILL ME
  LAST_SIGNAL
};

enum
{
  PROP_0,
  PROP_SILENT
};

// the capabilities of the inputs and outputs.
//
// describe the real formats here.
//
static GstStaticPadTemplate sink_factory =
    GST_STATIC_PAD_TEMPLATE(
        "sink",
        GST_PAD_SINK,
        GST_PAD_ALWAYS,
        GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE("RGB"))
    );

static GstStaticPadTemplate src_factory =
    GST_STATIC_PAD_TEMPLATE(
        "src",
        GST_PAD_SRC,
        GST_PAD_ALWAYS,
        GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE("RGB"))
    );

#define gst_my_npp_filter_parent_class parent_class

static void gst_my_npp_filter_set_property(GObject* object, guint prop_id,
    const GValue* value, GParamSpec* pspec);
static void gst_my_npp_filter_get_property(GObject* object, guint prop_id,
    GValue* value, GParamSpec* pspec);

static GstFlowReturn gst_my_npp_filter_transform_ip(GstBaseTransform* base,
    GstBuffer* outbuf);

static gboolean gst_my_npp_filter_set_caps(GstBaseTransform* base,
    GstCaps* incaps, GstCaps* outcaps);

static gboolean gst_my_npp_filter_start(GstBaseTransform* base);
static gboolean gst_my_npp_filter_stop(GstBaseTransform* base);

// GObject vmethod implementations

// initialize the myfilter's class
static void gst_my_npp_filter_class_init(GstMyNppFilterClass* klass) {
  GObjectClass* gobject_class;
  GstElementClass* gstelement_class;

  gobject_class = (GObjectClass*)klass;
  gstelement_class = (GstElementClass*)klass;

  gobject_class->set_property = gst_my_npp_filter_set_property;
  gobject_class->get_property = gst_my_npp_filter_get_property;

  g_object_class_install_property(gobject_class, PROP_SILENT,
      g_param_spec_boolean("silent", "Silent", "Produce verbose output ?",
          FALSE, G_PARAM_READWRITE
      )
  );

  gst_element_class_set_details_simple(gstelement_class,
      "MyNppFilter",
      "First NPP Test Filter",
      "First NPP Test Filter Element",
      "Marco Walther <marco@sonic.net>"
  );

  gst_element_class_add_pad_template(gstelement_class,
      gst_static_pad_template_get(&src_factory)
  );
  gst_element_class_add_pad_template(gstelement_class,
      gst_static_pad_template_get(&sink_factory)
  );

  GST_BASE_TRANSFORM_CLASS(klass)->transform_ip =
      GST_DEBUG_FUNCPTR(gst_my_npp_filter_transform_ip);
  GST_BASE_TRANSFORM_CLASS(klass)->set_caps =
      GST_DEBUG_FUNCPTR(gst_my_npp_filter_set_caps);
  GST_BASE_TRANSFORM_CLASS(klass)->start =
      GST_DEBUG_FUNCPTR(gst_my_npp_filter_start);
  GST_BASE_TRANSFORM_CLASS(klass)->stop =
      GST_DEBUG_FUNCPTR(gst_my_npp_filter_stop);

  // debug category for filtering log messages
  GST_DEBUG_CATEGORY_INIT(gst_my_npp_filter_debug, "plugin", 0, "My NPP Filter Plugin");
}

// initialize the new element
// instantiate pads and add them to element
// set pad callback functions
// initialize instance structure
static void
gst_my_npp_filter_init(GstMyNppFilter* filter) {
  const NppLibraryVersion* libVer = nppGetLibVersion();

  g_print("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

  int driverVersion;
  int runtimeVersion;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);

  g_print("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
      (driverVersion % 100) / 10);
  g_print("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
      (runtimeVersion % 100) / 10);

  // Min spec is SM 1.0 devices
  g_print("  CUDA Cap >= 1.0 : %d\n", static_cast<int>(checkCudaCapabilities(1, 0)));

  filter->silent = FALSE;
}

static void
gst_my_npp_filter_set_property(GObject* object, guint prop_id,
    const GValue* value, GParamSpec* pspec) {
  GstMyNppFilter* filter = GST_GST_MY_NPP_FILTER(object);

  switch (prop_id) {
    case PROP_SILENT:
      filter->silent = g_value_get_boolean(value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

static void
gst_my_npp_filter_get_property(GObject* object, guint prop_id,
    GValue* value, GParamSpec* pspec) {
  GstMyNppFilter* filter = GST_GST_MY_NPP_FILTER(object);

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean(value, filter->silent);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

// Called when source / sink pad capabilities have been negotiated.
static gboolean
gst_my_npp_filter_set_caps(GstBaseTransform* base, GstCaps* incaps,
    GstCaps* outcaps) {
  GstMyNppFilter* filter = GST_GST_MY_NPP_FILTER(base);

  // Save the input video information, since this will be required later.
  gst_video_info_from_caps(&filter->video_info, incaps);

  g_print("Video Info: %.16s (%d, %d) size: %lu\n",
      filter->video_info.finfo->name, filter->video_info.width,
      filter->video_info.height, filter->video_info.size);

  filter->npp_image11 =
    new npp::ImageNPP_8u_C1(filter->video_info.width, filter->video_info.height);
  filter->npp_image12 =
    new npp::ImageNPP_8u_C1(filter->video_info.width, filter->video_info.height);
  filter->npp_image13 =
    new npp::ImageNPP_8u_C1(filter->video_info.width, filter->video_info.height);
  filter->npp_image31 =
    new npp::ImageNPP_8u_C3(filter->video_info.width, filter->video_info.height);
  filter->npp_image32 =
    new npp::ImageNPP_8u_C3(filter->video_info.width, filter->video_info.height);
  filter->npp_image33 =
    new npp::ImageNPP_8u_C3(filter->video_info.width, filter->video_info.height);

  if (filter->npp_image11 == nullptr || filter->npp_image12 == nullptr || filter->npp_image13 == nullptr ||
      filter->npp_image31 == nullptr || filter->npp_image32 == nullptr || filter->npp_image33 == nullptr) {
    return FALSE;
  }

  NppiSize size_ROI = { filter->video_info.width, filter->video_info.height };
  int hpBufferSize1;
  int hpBufferSize2;

  NPP_CHECK_NPP(nppiLabelMarkersUFGetBufferSize_32u_C1R(size_ROI, &hpBufferSize1));
  NPP_CHECK_NPP(nppiCompressMarkerLabelsGetBufferSize_32u_C1R(size_ROI.width * size_ROI.height, &hpBufferSize2));
  if (hpBufferSize2 > hpBufferSize1) {
    hpBufferSize1 = hpBufferSize2;
  }
  cudaMalloc(&filter->label_maker_buffer, hpBufferSize1);

  cudaMalloc(&filter->label_maker_dst, size_ROI.width * size_ROI.height * sizeof(Npp32u));

  unsigned int hpBufferSize3;
  NPP_CHECK_NPP(nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R(101, &hpBufferSize3));
  g_print("filter->label_info_buffer size= %d\n", hpBufferSize3);
  cudaMalloc(&filter->label_info_buffer, hpBufferSize3);

  if (filter->label_maker_buffer == nullptr || filter->label_maker_dst == nullptr ||
      filter->label_info_buffer == nullptr) {
    return FALSE;
  }

  return TRUE;
}

static gboolean
gst_my_npp_filter_start(GstBaseTransform* base) {
  GstMyNppFilter* filter = GST_GST_MY_NPP_FILTER(base);

  cudaStreamCreate(&filter->cuda_stream);
  filter->npp_ctx.hStream = filter->cuda_stream; // Application-managed stream

  return TRUE;
}

static gboolean
gst_my_npp_filter_stop(GstBaseTransform* base) {
  GstMyNppFilter* filter = GST_GST_MY_NPP_FILTER(base);

  if (filter->npp_image11 != nullptr) {
    delete filter->npp_image11;
    filter->npp_image11 = nullptr;
  }

  if (filter->npp_image12 != nullptr) {
    delete filter->npp_image12;
    filter->npp_image12 = nullptr;
  }

  if (filter->npp_image13 != nullptr) {
    delete filter->npp_image13;
    filter->npp_image13 = nullptr;
  }

  if (filter->npp_image31 != nullptr) {
    delete filter->npp_image31;
    filter->npp_image31 = nullptr;
  }

  if (filter->npp_image32 != nullptr) {
    delete filter->npp_image32;
    filter->npp_image32 = nullptr;
  }

  if (filter->npp_image33 != nullptr) {
    delete filter->npp_image33;
    filter->npp_image33 = nullptr;
  }

  if (filter->cuda_stream != nullptr) {
    cudaStreamDestroy(filter->cuda_stream);
    filter->npp_ctx.hStream = nullptr;
    filter->cuda_stream = nullptr;
  }

  if (filter->label_maker_buffer != nullptr) {
    cudaFree(filter->label_maker_buffer);
    filter->label_maker_buffer = nullptr;
  }

  if (filter->label_maker_dst != nullptr) {
    cudaFree(filter->label_maker_dst);
    filter->label_maker_dst = nullptr;
  }

  if (filter->label_info_buffer != nullptr) {
    cudaFree(filter->label_info_buffer);
    filter->label_info_buffer = nullptr;
  }

  return TRUE;
}

// GstBaseTransform vmethod implementations

// this function does the actual processing
static GstFlowReturn
gst_my_npp_filter_transform_ip(GstBaseTransform* base, GstBuffer* outbuf) {
  GstMyNppFilter* filter = GST_GST_MY_NPP_FILTER(base);
  GstMapInfo in_map_info;

  if (GST_CLOCK_TIME_IS_VALID(GST_BUFFER_TIMESTAMP(outbuf))) {
    gst_object_sync_values(GST_OBJECT(filter), GST_BUFFER_TIMESTAMP(outbuf));
  }

  filter->frame_num++;

  memset(&in_map_info, 0, sizeof(in_map_info));
  if (!gst_buffer_map(outbuf, &in_map_info, GST_MAP_WRITE)) {
    g_print("Error: Failed to map gst buffer\n");
    return GST_FLOW_ERROR;
  }

  filter->npp_image31->copyFrom(in_map_info.data, filter->video_info.width * 3);

  NppiSize size_ROI = { filter->video_info.width, filter->video_info.height };
  NppiPoint offset = { 0, 0 };
  NppiSize mask_size = { 3, 3 };
  NppiPoint anchor = { mask_size.width / 2, mask_size.height / 2 };
  Npp8u thresholdArray[3];

  // Trying to implement something like
  // https://github.com/mw46d/Entdecker/blob/master/ROS/mw/mw_video/nodes/image_converter.py#L88 ff

  // npp_image31 is the RGB image
  // npp_image32 will be the the box filtered image
  NPP_CHECK_NPP(nppiFilterBoxBorder_8u_C3R_Ctx(filter->npp_image31->data(), filter->npp_image31->pitch(),
      size_ROI, offset,
      filter->npp_image32->data(), filter->npp_image32->pitch(),
      size_ROI, mask_size, anchor, NPP_BORDER_REPLICATE, filter->npp_ctx));

  // npp_image32 is the filtered source image
  // npp_image33 will be the HSV image to work on
  NPP_CHECK_NPP(nppiRGBToHSV_8u_C3R_Ctx(filter->npp_image32->data(), filter->npp_image32->pitch(),
      filter->npp_image33->data(), filter->npp_image33->pitch(),
      size_ROI, filter->npp_ctx));

  // First Threshold check
  // npp_image33 is the filtered HSV source image
  // npp_image11 will be the mat with markes where the pixels are >= (0, 170, 125)
  thresholdArray[0] = 0;
  thresholdArray[1] = 170;
  thresholdArray[2] = 125;

  NPP_CHECK_NPP(nppiCompareC_8u_C3R_Ctx(filter->npp_image33->data(), filter->npp_image33->pitch(),
      thresholdArray,
      filter->npp_image11->data(), filter->npp_image11->pitch(),
      size_ROI, NPP_CMP_GREATER_EQ, filter->npp_ctx));

  // npp_image33 is the filtered HSV source image
  // npp_image12 will be the mat with markes where the pixels are <= (10, 255, 255)
  thresholdArray[0] = 10;
  thresholdArray[1] = 255;
  thresholdArray[2] = 255;

  NPP_CHECK_NPP(nppiCompareC_8u_C3R_Ctx(filter->npp_image33->data(), filter->npp_image33->pitch(),
      thresholdArray,
      filter->npp_image12->data(), filter->npp_image12->pitch(),
      size_ROI, NPP_CMP_LESS_EQ, filter->npp_ctx));

  // Combine the two sides of the threshold check
  // npp_image11 &&= npp_image12
  NPP_CHECK_NPP(nppiAnd_8u_C1IR_Ctx(filter->npp_image12->data(), filter->npp_image12->pitch(),
      filter->npp_image11->data(), filter->npp_image11->pitch(),
      size_ROI, filter->npp_ctx));

  // Second Threshold check
  // npp_image33 is the filtered HSV source image
  // npp_image12 will be the mat with markes where the pixels are >= (170, 170, 125)
  thresholdArray[0] = 170; 
  thresholdArray[1] = 170;
  thresholdArray[2] = 125;

  NPP_CHECK_NPP(nppiCompareC_8u_C3R_Ctx(filter->npp_image33->data(), filter->npp_image33->pitch(), 
      thresholdArray, 
      filter->npp_image12->data(), filter->npp_image12->pitch(),
      size_ROI, NPP_CMP_GREATER_EQ, filter->npp_ctx));
 
  // npp_image33 is the filtered HSV source image
  // npp_image13 will be the mat with markes where the pixels are <= (180, 255, 255)
  thresholdArray[0] = 180;
  thresholdArray[1] = 255;
  thresholdArray[2] = 255;

  NPP_CHECK_NPP(nppiCompareC_8u_C3R_Ctx(filter->npp_image33->data(), filter->npp_image33->pitch(),
      thresholdArray,
      filter->npp_image13->data(), filter->npp_image13->pitch(),
      size_ROI, NPP_CMP_LESS_EQ, filter->npp_ctx));

  // Combine the two sides of the threshold check
  // npp_image12 &&= npp_image13
  NPP_CHECK_NPP(nppiAnd_8u_C1IR_Ctx(filter->npp_image13->data(), filter->npp_image13->pitch(),
      filter->npp_image12->data(), filter->npp_image12->pitch(),
      size_ROI, filter->npp_ctx));

  // Combine both thresholds
  // npp_image11 ||= npp_image12
  NPP_CHECK_NPP(nppiOr_8u_C1IR_Ctx(filter->npp_image12->data(), filter->npp_image12->pitch(),
      filter->npp_image11->data(), filter->npp_image11->pitch(),
      size_ROI, filter->npp_ctx));

  // Temp!!!!
  unsigned char* tmp_buffer = static_cast<unsigned char*>(alloca(filter->video_info.width * filter->video_info.height * sizeof(char)));
  filter->npp_image11->copyTo(tmp_buffer, filter->video_info.width);

  // Just a small part of the image
  for (int i = 0; i < filter->video_info.width * filter->video_info.height; i++) {
    int ow = (i / filter->video_info.width) / 3;
    int oh = (i % filter->video_info.width) / 3;
    in_map_info.data[(ow * filter->video_info.width + oh) * 3 + 0 ] = tmp_buffer[i];
    in_map_info.data[(ow * filter->video_info.width + oh) * 3 + 1 ] = tmp_buffer[i];
    in_map_info.data[(ow * filter->video_info.width + oh) * 3 + 2 ] = tmp_buffer[i];
  }

  NPP_CHECK_NPP(nppiLabelMarkersUF_8u32u_C1R_Ctx(filter->npp_image11->data(), filter->npp_image11->pitch(),
      filter->label_maker_dst, size_ROI.width * sizeof(Npp32u),
      size_ROI, nppiNormL1, filter->label_maker_buffer, filter->npp_ctx));

  // XXX Recycle the buffer area!
  int max_num = 0;
  NPP_CHECK_NPP(nppiCompressMarkerLabelsUF_32u_C1IR_Ctx(filter->label_maker_dst, size_ROI.width * sizeof(Npp32u),
      size_ROI, size_ROI.width * size_ROI.height, &max_num, filter->label_maker_buffer, filter->npp_ctx));

  g_print("mw t2a max_num= %d", max_num);
  if (max_num > 100) {
    max_num = 100;
  }

  // XXX Optional does not really seem to be optional:-( 
  Npp8u* d_contours = nullptr; cudaMalloc(&d_contours, size_ROI.width * size_ROI.height);
  NppiContourPixelDirectionInfo* d_directions = nullptr; cudaMalloc(&d_directions, size_ROI.width * size_ROI.height * sizeof(NppiContourPixelDirectionInfo));
  NppiContourTotalsInfo contoursTotalsInfoHost;
  Npp32u* d_counts = nullptr; cudaMalloc(&d_counts, (max_num + 4) * sizeof(Npp32u));
  Npp32u* contoursPixelCountsListHost = static_cast<Npp32u*>(alloca((max_num  + 4) * sizeof(Npp32u)));
  Npp32u* d_found = nullptr; cudaMalloc(&d_found, (max_num + 4) * sizeof(Npp32u));
  Npp32u* contoursPixelStartingOffsetHost = static_cast<Npp32u*>(alloca((max_num  + 4) * sizeof(Npp32u)));
  Npp32u* d_offsets = nullptr; cudaMalloc(&d_offsets, (max_num + 4) * sizeof(Npp32u));

  NPP_CHECK_NPP(nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(filter->label_maker_dst, size_ROI.width * sizeof(Npp32u),
      size_ROI, max_num, filter->label_info_buffer,
      d_contours, size_ROI.width, d_directions, size_ROI.width * sizeof(NppiContourPixelDirectionInfo),
      &contoursTotalsInfoHost, d_counts, contoursPixelCountsListHost, d_offsets, contoursPixelStartingOffsetHost,
      filter->npp_ctx)); 

  cudaStreamSynchronize(filter->npp_ctx.hStream);
  cudaFree(d_contours);
  cudaFree(d_directions);
  cudaFree(d_counts);
  cudaFree(d_found);
  cudaFree(d_offsets);
/*  
  NPP_CHECK_NPP(nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(filter->label_maker_dst, size_ROI.width * sizeof(Npp32u),
      size_ROI, max_num, label_info_buffer,
      NULL, 0, NULL, 0,
      NULL, NULL, NULL, NULL, NULL,
      filter->npp_ctx));
*/
  NppiCompressedMarkerLabelsInfo* host_info_list = static_cast<NppiCompressedMarkerLabelsInfo*>(alloca(max_num * sizeof(NppiCompressedMarkerLabelsInfo)));

  cudaMemcpy(host_info_list, filter->label_info_buffer, max_num * sizeof(NppiCompressedMarkerLabelsInfo), cudaMemcpyDeviceToHost);

  for (int i = 0; i < max_num; i++) {
    g_print("MW bounding_box[%u] = (%d, %d, %d, %d)\n", i, host_info_list[i].oMarkerLabelBoundingBox.x, host_info_list[i].oMarkerLabelBoundingBox.y, host_info_list[i].oMarkerLabelBoundingBox.width, host_info_list[i].oMarkerLabelBoundingBox.height);
    for (int x = host_info_list[i].oMarkerLabelBoundingBox.x; x < host_info_list[i].oMarkerLabelBoundingBox.width; x++) {
      in_map_info.data[(host_info_list[i].oMarkerLabelBoundingBox.y * size_ROI.width + x) * 3] = 255;
      in_map_info.data[(host_info_list[i].oMarkerLabelBoundingBox.height* size_ROI.width + x) * 3] = 255;
    }
    for (int y = host_info_list[i].oMarkerLabelBoundingBox.y; y < host_info_list[i].oMarkerLabelBoundingBox.height; y++) {
      in_map_info.data[(host_info_list[i].oMarkerLabelBoundingBox.x + filter->video_info.width * y) * 3] = 255;
      in_map_info.data[(host_info_list[i].oMarkerLabelBoundingBox.width + filter->video_info.width * y) * 3] = 255;
    }
  }

  return GST_FLOW_OK;
}


// entry point to initialize the plug-in
// initialize the plug-in itself
// register the element factories and other features
static gboolean
plugin_init(GstPlugin* myfilter) {
  GST_DEBUG_CATEGORY_INIT(gst_my_npp_filter_debug, "mynppfilter",
      0, "Test mynppfilter");

  return gst_element_register(myfilter, "mynppfilter", GST_RANK_NONE,
      GST_TYPE_MY_NPP_FILTER);
}

// PACKAGE: this is usually set by autotools depending on some _INIT macro
// in configure.ac and then written into and defined in config.h, but we can
// just set it ourselves here in case someone doesn't use autotools to
// compile this code. GST_PLUGIN_DEFINE needs PACKAGE to be defined.
//
#ifndef PACKAGE
#define PACKAGE "mynppfilter"
#endif

// gstreamer looks for this structure to register mynppfilter
GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    mynppfilter,
    "Test mynppfilter",
    plugin_init,
    "0.1.0",
    "LGPL",
    "GStreamer",
    "http://gstreamer.net/"
)
