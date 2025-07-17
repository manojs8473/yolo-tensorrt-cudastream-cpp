#pragma once

#include "NvInfer.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include "detection_types.h"

using namespace nvinfer1;
using namespace std;
using namespace cv;

class YOLOv11
{

public:

    YOLOv11(string model_path, nvinfer1::ILogger& logger, float conf_threshold = 0.5f);
    ~YOLOv11();

    void preprocess(Mat& image);
    void infer();
    void postprocess(vector<Detection>& output);
    void postprocess_start_next_copy(); // Start copy for next frame
    void draw(Mat& image, const vector<Detection>& output);
    void drawTimingInfo(Mat& image);
    
    // Timing getters
    float getPreprocessTime() const { return preprocess_time; }
    float getInferenceTime() const { return inference_time; }
    float getPostprocessTime() const { return postprocess_time; }
    float getTotalTime() const { return total_time; }

private:
    void init(std::string engine_path, nvinfer1::ILogger& logger);

    float* gpu_buffers[2];               //!< The vector of device buffers needed for engine execution
    float* cpu_output_buffer;
    
    // GPU-side filtering buffers
    FilteredDetection* filtered_detections_host;
    int detection_count;
    
    // Streaming/overlapping optimization
    float* cpu_output_buffer_alt;        //!< Double buffer for overlapping
    cudaStream_t stream;
    cudaStream_t stream_alt;             //!< Second stream for overlapping
    int current_buffer_idx;              //!< 0 or 1 for double buffering
    IRuntime* runtime;                 //!< The TensorRT runtime used to deserialize the engine
    ICudaEngine* engine;               //!< The TensorRT engine used to run the network
    IExecutionContext* context;        //!< The context for executing inference using an ICudaEngine

    // Model parameters
    int input_w;
    int input_h;
    int num_detections;
    int detection_attribute_size;
    int num_classes = 1;
    const int MAX_IMAGE_SIZE = 4096 * 4096;
    float conf_threshold = 0.5f;
    float nms_threshold = 0.4f;

    vector<Scalar> colors;

    // Timing variables
    float preprocess_time;
    float inference_time;
    float postprocess_time;
    float total_time;
    
    // Rolling average for display (10 frames)
    static const int ROLLING_WINDOW = 10;
    float preprocess_history[ROLLING_WINDOW];
    float inference_history[ROLLING_WINDOW];
    float postprocess_history[ROLLING_WINDOW];
    float total_history[ROLLING_WINDOW];
    int history_index;

    void build(std::string onnxPath, nvinfer1::ILogger& logger);
    bool saveEngine(const std::string& filename);
};