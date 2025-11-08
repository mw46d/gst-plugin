#ifndef __DSEXAMPLE__
#define __DSEXAMPLE__

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaarithm.hpp"

#define MAX_LABEL_SIZE 128
#ifdef __cplusplus
extern "C" {
#endif

typedef struct DsExampleCtx DsExampleCtx;

// Init parameters structure as input, required for instantiating dsexample_lib
typedef struct {
  // Width at which frame/object will be scaled
  int processingWidth;
  // height at which frame/object will be scaled
  int processingHeight;

} DsExampleInitParams;

// Detected/Labelled object structure, stores bounding box info along with label
typedef struct {
  float left;
  float top;
  float width;
  float height;
  char label[MAX_LABEL_SIZE];
} DsExampleObject;

// Output data returned after processing
typedef struct {
  int numObjects;
  DsExampleObject object[4];
} DsExampleOutput;

// Initialize library context
DsExampleCtx * DsExampleCtxInit (DsExampleInitParams *init_params);

// Dequeue processed output
DsExampleOutput *DsExampleProcess (DsExampleCtx *ctx, cv::cuda::GpuMat* inmat);

// Deinitialize library context
void DsExampleCtxDeinit (DsExampleCtx *ctx);

#ifdef __cplusplus
}
#endif

#endif
