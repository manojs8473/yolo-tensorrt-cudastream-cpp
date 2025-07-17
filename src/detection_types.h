#pragma once

#include <opencv2/opencv.hpp>

using namespace cv;

struct Detection
{
    float conf;
    int class_id;
    Rect bbox;
};

// GPU-side confidence filtering
struct FilteredDetection {
    float cx, cy, w, h;
    float confidence;
    int class_id;
};