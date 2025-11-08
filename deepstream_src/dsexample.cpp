#include "dsexample.h"
#include <stdio.h>
#include <stdlib.h>
#include <gst/video/video.h>

struct DsExampleCtx {
    DsExampleInitParams initParams;
};

DsExampleCtx*
DsExampleCtxInit (DsExampleInitParams * initParams) {
    DsExampleCtx* ctx = (DsExampleCtx*)calloc(1, sizeof(DsExampleCtx));
    ctx->initParams = *initParams;

    return ctx;
}

DsExampleOutput *
DsExampleProcess(DsExampleCtx* ctx, cv::cuda::GpuMat* inmat) {
    DsExampleOutput* out = nullptr;

    cv::Ptr<cv::cuda::Filter> gaussian_filter = cv::cuda::createGaussianFilter(
        CV_8UC3, CV_8UC3, cv::Size(5, 5), 0);

    cv::cuda::GpuMat blur(inmat->size(), CV_8UC3);
    gaussian_filter->apply(*inmat, blur);

    cv::cuda::GpuMat hsv(inmat->size(), CV_8UC3);
    cv::cuda::cvtColor(blur, hsv, cv::COLOR_BGR2HSV);

    cv::cuda::GpuMat threshold(inmat->size(), CV_8UC1);
    cv::cuda::GpuMat threshold1(inmat->size(), CV_8UC1);
    cv::cuda::GpuMat threshold2(inmat->size(), CV_8UC1);
    cv::cuda::inRange(hsv, cv::Scalar(0, 170, 125, 0), cv::Scalar(10, 255, 255, 0), threshold1);
    cv::cuda::inRange(hsv, cv::Scalar(170, 170, 125, 0), cv::Scalar(180, 255, 255, 0), threshold2);

    cv::cuda::bitwise_or(threshold1, threshold2, threshold);
    cv::Mat h_threshold(threshold.size(), CV_8UC1);
    threshold.download(h_threshold);

#ifdef MW_DEBUG
    cv::Mat h_temp(hsv.size(), CV_8UC3);
    hsv.download(h_temp);

    // Just a small part of the image
    int w = threshold.size().width;
    int h = threshold.size().height;
    for (int i = 0; i < w * h; i++) {
        int ow = (i / w) / 3;
        int oh = (i % w) / 3;
        h_temp.data[(ow * w + oh) * 3 + 0 ] = h_threshold.data[i];
        h_temp.data[(ow * w + oh) * 3 + 1 ] = h_threshold.data[i];
        h_temp.data[(ow * w + oh) * 3 + 2 ] = h_threshold.data[i];
    }

    g_print("MW h_temp = %d x %d\n", h_temp.cols, h_temp.rows);
    cv::imshow("Test", h_temp);
#endif

    std::vector< std::vector< cv::Point> > h_contours;

    cv::findContours(h_threshold, h_contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    DsExampleObject* bboxes = static_cast<DsExampleObject*>(alloca((h_contours.size() + 1) * sizeof(DsExampleObject)));
    int num_bboxes = 0;

    for (int i = 0; i< h_contours.size(); i++) {
        int x1 = inmat->size().width;
        int x2 = 0;
        int y1 = inmat->size().height;
        int y2 = 0;

        for (int k = 0; k < h_contours[i].size(); k++) {
            if (h_contours[i][k].x < x1) {
                x1 = h_contours[i][k].x;
            }
            if (h_contours[i][k].y < y1) {
                y1 = h_contours[i][k].y;
            }
            if (h_contours[i][k].x > x2) {
                x2 = h_contours[i][k].x;
            }
            if (h_contours[i][k].y > y2) {
                y2 = h_contours[i][k].y;
            }
        }

        float height = y2 - y1;
        float width = x2 - x1;
        if (width * height > 320 && width / height < 1.0) {
            bboxes[num_bboxes].left = (float)x1;
            bboxes[num_bboxes].top = (float)y1;
            bboxes[num_bboxes].width = width;
            bboxes[num_bboxes].height = height;
            snprintf(bboxes[num_bboxes].label, 64, "Cone %d", num_bboxes);
            num_bboxes++;
        }
    }

#ifdef MW_DEBUG
    bboxes[num_bboxes].left = (float)(ctx->initParams.processingWidth) / 8;
    bboxes[num_bboxes].top = (float)(ctx->initParams.processingHeight) / 8;
    bboxes[num_bboxes].width = (float)(ctx->initParams.processingWidth) / 8;
    bboxes[num_bboxes].height = (float)(ctx->initParams.processingHeight) / 8;
    snprintf(bboxes[num_bboxes].label, 64, "Test %d  ", num_bboxes);
    num_bboxes++;

    g_print("MW h_contours.size()= %ld num_bboxes= %d\n", h_contours.size(), num_bboxes);
#endif

    out = (DsExampleOutput*)calloc(1, sizeof(DsExampleOutput) + (num_bboxes - 4) * sizeof(DsExampleObject));
    out->numObjects = num_bboxes;
    memcpy((unsigned char *)out + sizeof(int), bboxes, sizeof(DsExampleObject) * num_bboxes);
#ifdef MW_DEBUG
    for (int i = 0; i < num_bboxes; i++) {
        g_print("MW bbox[%d] = %s (%f, %f, %f, %f)\n", i, out->object[i].label, \
            out->object[i].left, out->object[i].top, out->object[i].width, out->object[i].height);
    }
#endif

    return out;
}

void
DsExampleCtxDeinit (DsExampleCtx* ctx) {
    free (ctx);
}
