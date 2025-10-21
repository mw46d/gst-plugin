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

#include <gst/gst.h>
#include <gst/base/base.h>
#include <gst/controller/controller.h>
#include <iostream>
#include "mynppfilter.h"


GST_DEBUG_CATEGORY_STATIC (gst_my_npp_filter_debug);
#define GST_CAT_DEFAULT gst_my_npp_filter_debug

/* Filter signals and args */
enum
{
  /* FILL ME */
  LAST_SIGNAL
};

enum
{
  PROP_0,
  PROP_SILENT
};

/* the capabilities of the inputs and outputs.
 *
 * describe the real formats here.
 */
static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE ("RGB"))
    );

static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE ("RGB"))
    );

#define gst_my_npp_filter_parent_class parent_class

static void gst_my_npp_filter_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_my_npp_filter_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static GstFlowReturn gst_my_npp_filter_transform_ip (GstBaseTransform * base,
    GstBuffer * outbuf);

static gboolean gst_my_npp_filter_set_caps (GstBaseTransform * base,
    GstCaps * incaps, GstCaps * outcaps);

static gboolean gst_my_npp_filter_start (GstBaseTransform * base);
static gboolean gst_my_npp_filter_stop (GstBaseTransform * base);

/* GObject vmethod implementations */

/* initialize the myfilter's class */
static void
gst_my_npp_filter_class_init (GstMyNppFilterClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;

  gobject_class->set_property = gst_my_npp_filter_set_property;
  gobject_class->get_property = gst_my_npp_filter_get_property;

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output ?",
          FALSE, G_PARAM_READWRITE));

  gst_element_class_set_details_simple(gstelement_class,
    "MyNppFilter",
    "First NPP Test Filter",
    "First NPP Test Filter Element",
    "Marco Walther <marco@sonic.net>");

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_factory));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_factory));

  GST_BASE_TRANSFORM_CLASS (klass)->transform_ip =
      GST_DEBUG_FUNCPTR (gst_my_npp_filter_transform_ip);

  GST_BASE_TRANSFORM_CLASS (klass)->set_caps =
      GST_DEBUG_FUNCPTR (gst_my_npp_filter_set_caps);
  GST_BASE_TRANSFORM_CLASS (klass)->start =
      GST_DEBUG_FUNCPTR (gst_my_npp_filter_start);
  GST_BASE_TRANSFORM_CLASS (klass)->stop =
      GST_DEBUG_FUNCPTR (gst_my_npp_filter_stop);

  /* debug category for fltering log messages
   */
  GST_DEBUG_CATEGORY_INIT (gst_my_npp_filter_debug, "plugin", 0, "My NPP Filter Plugin");
}

/* initialize the new element
 * instantiate pads and add them to element
 * set pad calback functions
 * initialize instance structure
 */
static void
gst_my_npp_filter_init (GstMyNppFilter * filter)
{
  const NppLibraryVersion *libVer = nppGetLibVersion();

  g_print("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
         libVer->build);

  int driverVersion, runtimeVersion;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);

  g_print("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
         (driverVersion % 100) / 10);
  g_print("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
         (runtimeVersion % 100) / 10);

  // Min spec is SM 1.0 devices
  bool bVal = checkCudaCapabilities(1, 0);
  g_print("  CUDA Cap >= 1.0 : %d\n", (int)bVal);

  filter->silent = FALSE;
}

static void
gst_my_npp_filter_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstMyNppFilter *filter = GST_GST_MY_NPP_FILTER (object);

  switch (prop_id) {
    case PROP_SILENT:
      filter->silent = g_value_get_boolean (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_my_npp_filter_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstMyNppFilter *filter = GST_GST_MY_NPP_FILTER (object);

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, filter->silent);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * Called when source / sink pad capabilities have been negotiated.
 */
static gboolean
gst_my_npp_filter_set_caps (GstBaseTransform * base, GstCaps * incaps,
    GstCaps * outcaps)
{
  GstMyNppFilter *filter = GST_GST_MY_NPP_FILTER (base);

  /* Save the input video information, since this will be required later. */
  gst_video_info_from_caps (&filter->video_info, incaps);

  g_print ("Video Info: %.16s (%d, %d) size: %ud\n", filter->video_info.finfo->name, filter->video_info.width, filter->video_info.height, (unsigned int)filter->video_info.size);

  filter->npp_image_p = new npp::ImageNPP_8u_C3(filter->video_info.width, filter->video_info.height);

  return TRUE;
}

static gboolean
gst_my_npp_filter_start (GstBaseTransform * base)
{
  GstMyNppFilter *filter = GST_GST_MY_NPP_FILTER (base);

  cudaStreamCreate(&filter->cuda_stream);
  filter->npp_ctx.hStream = filter->cuda_stream; // Application-managed stream

  return TRUE;
}

static gboolean
gst_my_npp_filter_stop (GstBaseTransform * base)
{
  GstMyNppFilter *filter = GST_GST_MY_NPP_FILTER (base);

  if (filter->npp_image_p != NULL) {
    delete filter->npp_image_p;
    filter->npp_image_p = NULL;
  }

  if (filter->cuda_stream != NULL) {
    cudaStreamDestroy (filter->cuda_stream);
    filter->npp_ctx.hStream = NULL;
    filter->cuda_stream = NULL;
  }

  return TRUE;
}

/* GstBaseTransform vmethod implementations */

/* this function does the actual processing
 */
static GstFlowReturn
gst_my_npp_filter_transform_ip (GstBaseTransform * base, GstBuffer * outbuf)
{
  GstMyNppFilter *filter = GST_GST_MY_NPP_FILTER (base);
  GstMapInfo in_map_info;

  if (GST_CLOCK_TIME_IS_VALID (GST_BUFFER_TIMESTAMP (outbuf))) {
    gst_object_sync_values (GST_OBJECT (filter), GST_BUFFER_TIMESTAMP (outbuf));
  }

  /*
  if (filter->silent == FALSE) {
    g_print ("Loaded!\n");
    // Now we can use iostream C++:
    std::cout<< "Test" <<std::endl;
  }
  */

  filter->frame_num++;

  memset (&in_map_info, 0, sizeof (in_map_info));
  if (!gst_buffer_map (outbuf, &in_map_info, GST_MAP_WRITE)) {
    g_print ("Error: Failed to map gst buffer\n");
    return GST_FLOW_ERROR;
  }

  // !!!! Add some 'random' values to the image !!!!
  filter->npp_image_p->copyFrom(in_map_info.data, filter->video_info.width * 3);
  NppiSize size_ROI = { filter->video_info.width, filter->video_info.height };
  const Npp8u constArray[] = { (Npp8u)(filter->frame_num % 180), (Npp8u)((filter->frame_num + 60) % 180), (Npp8u)((filter->frame_num + 120) % 180) };
  // g_print ("MW [ %d, %d, %d ]\n", constArray[0], constArray[1], constArray[2]);

  nppiAddC_8u_C3IRSfs_Ctx(constArray, filter->npp_image_p->data(), filter->npp_image_p->pitch(), size_ROI, 1, filter->npp_ctx);

  filter->npp_image_p->copyTo(in_map_info.data, filter->video_info.width * 3);

  return GST_FLOW_OK;

}


/* entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
static gboolean
mynppfilter_init (GstPlugin * myfilter)
{
  /* debug category for fltering log messages
   *
   * exchange the string 'Template myfilter' with your description
   */
  GST_DEBUG_CATEGORY_INIT (gst_my_npp_filter_debug, "mynppfilter",
      0, "Test mynppfilter");

  return gst_element_register (myfilter, "mynppfilter", GST_RANK_NONE,
      GST_TYPE_MY_NPP_FILTER);
}

/* PACKAGE: this is usually set by autotools depending on some _INIT macro
 * in configure.ac and then written into and defined in config.h, but we can
 * just set it ourselves here in case someone doesn't use autotools to
 * compile this code. GST_PLUGIN_DEFINE needs PACKAGE to be defined.
 */
#ifndef PACKAGE
#define PACKAGE "mynppfilter"
#endif

/* gstreamer looks for this structure to register myfilters
 *
 * exchange the string 'Template myfilter' with your myfilter description
 */
GST_PLUGIN_DEFINE (
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    mynppfilter,
    "Test mynppfilter",
    mynppfilter_init,
    "0.1.0",
    "LGPL",
    "GStreamer",
    "http://gstreamer.net/"
)
