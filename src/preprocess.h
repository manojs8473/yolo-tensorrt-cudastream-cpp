#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include "detection_types.h"

void cuda_preprocess_init(int max_image_size);
void cuda_preprocess_destroy();
void cuda_preprocess(uint8_t* src, int src_width, int src_height,
    float* dst, int dst_width, int dst_height,
    cudaStream_t stream);

void cuda_postprocess_init(int max_detections);
void cuda_postprocess_destroy();
int cuda_confidence_filter(float* raw_detections, int num_detections, int num_classes, 
                          float conf_threshold, FilteredDetection* filtered_output, 
                          int* detection_count, cudaStream_t stream);