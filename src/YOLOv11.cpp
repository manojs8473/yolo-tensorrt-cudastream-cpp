#include "YOLOv11.h"
#include "logging.h"
#include "cuda_utils.h"
#include "macros.h"
#include "preprocess.h"
#include <NvOnnxParser.h>
#include "common.h"
#include <fstream>
#include <iostream>


static Logger logger;
#define isFP16 true
#define warmup true


YOLOv11::YOLOv11(string model_path, nvinfer1::ILogger& logger, float conf_threshold)
{
    this->conf_threshold = conf_threshold;
    
    // Deserialize an engine
    if (model_path.find(".onnx") == std::string::npos)
    {
        init(model_path, logger);
    }
    // Build an engine from an onnx model
    else
    {
        build(model_path, logger);
        saveEngine(model_path);
    }

#if NV_TENSORRT_MAJOR < 10
    // Define input dimensions
    auto input_dims = engine->getBindingDimensions(0);
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
#else
    auto input_dims = engine->getTensorShape(engine->getIOTensorName(0));
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
#endif
}


void YOLOv11::init(std::string engine_path, nvinfer1::ILogger& logger)
{
    // Read the engine file
    ifstream engineStream(engine_path, ios::binary);
    engineStream.seekg(0, ios::end);
    const size_t modelSize = engineStream.tellg();
    engineStream.seekg(0, ios::beg);
    unique_ptr<char[]> engineData(new char[modelSize]);
    engineStream.read(engineData.get(), modelSize);
    engineStream.close();

    // Deserialize the tensorrt engine
    runtime = createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(engineData.get(), modelSize);
    context = engine->createExecutionContext();

    // Get input and output sizes of the model
// Get the number of bindings (inputs + outputs)
    int32_t num_bindings = engine->getNbIOTensors();

    // Get input dimensions (assuming first tensor is input)
    const char* input_name = engine->getIOTensorName(0);
    nvinfer1::Dims input_dims = engine->getTensorShape(input_name);
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];

    // Get output dimensions (assuming second tensor is output)
    const char* output_name = engine->getIOTensorName(1);
    nvinfer1::Dims output_dims = engine->getTensorShape(output_name);
    detection_attribute_size = output_dims.d[1];
    num_detections = output_dims.d[2];
    num_classes = detection_attribute_size - 4;

    // Initialize input buffers with pinned memory for faster GPU-CPU transfers
    CUDA_CHECK(cudaHostAlloc((void**)&cpu_output_buffer, detection_attribute_size * num_detections * sizeof(float), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&cpu_output_buffer_alt, detection_attribute_size * num_detections * sizeof(float), cudaHostAllocDefault));
    CUDA_CHECK(cudaMalloc(&gpu_buffers[0], 3 * input_w * input_h * sizeof(float)));
    // Initialize output buffer
    CUDA_CHECK(cudaMalloc(&gpu_buffers[1], detection_attribute_size * num_detections * sizeof(float)));

    cuda_preprocess_init(MAX_IMAGE_SIZE);
    cuda_postprocess_init(num_detections);
    
    // Initialize host buffer for filtered detections
    CUDA_CHECK(cudaHostAlloc((void**)&filtered_detections_host, num_detections * sizeof(FilteredDetection), cudaHostAllocDefault));

    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaStreamCreate(&stream_alt));
    
    current_buffer_idx = 0;
    history_index = 0;
    
    // Initialize timing history arrays
    for (int i = 0; i < ROLLING_WINDOW; i++) {
        preprocess_history[i] = 0.0f;
        inference_history[i] = 0.0f;
        postprocess_history[i] = 0.0f;
        total_history[i] = 0.0f;
    }


    if (warmup) {
        for (int i = 0; i < 10; i++) {
            this->infer();
        }
        printf("model warmup 10 times\n");
    }
}

YOLOv11::~YOLOv11()
{
    // Release stream and buffers
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamSynchronize(stream_alt));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaStreamDestroy(stream_alt));
    for (int i = 0; i < 2; i++)
        CUDA_CHECK(cudaFree(gpu_buffers[i]));
    CUDA_CHECK(cudaFreeHost(cpu_output_buffer));
    CUDA_CHECK(cudaFreeHost(cpu_output_buffer_alt));
    CUDA_CHECK(cudaFreeHost(filtered_detections_host));

    // Destroy the engine
    cuda_preprocess_destroy();
    cuda_postprocess_destroy();
    delete context;
    delete engine;
    delete runtime;
}

void YOLOv11::preprocess(Mat& image) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Store original image dimensions for scaling in postprocess
    original_image_width = image.cols;
    original_image_height = image.rows;
    
    // Preprocessing data on gpu
    cuda_preprocess(image.ptr(), image.cols, image.rows, gpu_buffers[0], input_w, input_h, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    auto end = std::chrono::high_resolution_clock::now();
    preprocess_time = std::chrono::duration<float, std::milli>(end - start).count();
}

void YOLOv11::infer()
{
    auto start = std::chrono::high_resolution_clock::now();
    
#if NV_TENSORRT_MAJOR < 10
    context->enqueueV2((void**)gpu_buffers, stream, nullptr);
#else
    const char* input_name = engine->getIOTensorName(0);
    const char* output_name = engine->getIOTensorName(1);
    context->setTensorAddress(input_name, gpu_buffers[0]);
    context->setTensorAddress(output_name, gpu_buffers[1]);
    this->context->enqueueV3(this->stream);
#endif
    
    auto end = std::chrono::high_resolution_clock::now();
    inference_time = std::chrono::duration<float, std::milli>(end - start).count();
}

void YOLOv11::postprocess(vector<Detection>& output)
{
    auto start = std::chrono::high_resolution_clock::now();
    
    // Streaming/overlapping optimization:
    // The memory copy was already started by postprocess_start_next_copy()
    // We just need to wait for it to complete
    float* current_cpu_buffer = (current_buffer_idx == 0) ? cpu_output_buffer_alt : cpu_output_buffer;
    cudaStream_t current_stream = (current_buffer_idx == 0) ? stream_alt : stream;
    
    // Wait for the copy that was started earlier to complete
    CUDA_CHECK(cudaStreamSynchronize(current_stream));
    
    auto memcpy_end = std::chrono::high_resolution_clock::now();
    float memcpy_time = std::chrono::duration<float, std::milli>(memcpy_end - start).count();

    // Pre-allocate vectors to avoid reallocations
    vector<Rect> boxes;
    vector<int> class_ids;
    vector<float> confidences;
    
    // Estimate ~10% of detections pass threshold for typical scenes
    int estimated_detections = num_detections / 10;
    boxes.reserve(estimated_detections);
    class_ids.reserve(estimated_detections);
    confidences.reserve(estimated_detections);

    // Direct pointer access - column-major format (detection_attribute_size x num_detections)
    const float* data = current_cpu_buffer;
    
    auto score_loop_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_detections; ++i) {
        // Manual argmax - optimized with direct pointer access
        float max_score = -1.0f;
        int max_class_id = -1;
        
        // Find class with highest score - column-major memory layout
        for (int j = 0; j < num_classes; ++j) {
            float score = data[(4 + j) * num_detections + i];
            if (score > max_score) {
                max_score = score;
                max_class_id = j;
            }
        }

        if (max_score > conf_threshold) {
            const float cx = data[0 * num_detections + i];
            const float cy = data[1 * num_detections + i];
            const float ow = data[2 * num_detections + i];
            const float oh = data[3 * num_detections + i];
            Rect box;
            box.x = static_cast<int>((cx - 0.5 * ow));
            box.y = static_cast<int>((cy - 0.5 * oh));
            box.width = static_cast<int>(ow);
            box.height = static_cast<int>(oh);

            boxes.push_back(box);
            class_ids.push_back(max_class_id);
            confidences.push_back(max_score);
        }
    }
    
    auto score_loop_end = std::chrono::high_resolution_clock::now();
    float score_loop_time = std::chrono::duration<float, std::milli>(score_loop_end - score_loop_start).count();

    auto nms_start = std::chrono::high_resolution_clock::now();
    vector<int> nms_result;
    dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_result);
    auto nms_end = std::chrono::high_resolution_clock::now();
    float nms_time = std::chrono::duration<float, std::milli>(nms_end - nms_start).count();

    // Calculate scaling ratios
    const float ratio_h = input_h / (float)original_image_height;
    const float ratio_w = input_w / (float)original_image_width;
    
    for (int i = 0; i < nms_result.size(); i++)
    {
        Detection result;
        int idx = nms_result[i];
        result.class_id = class_ids[idx];
        result.conf = confidences[idx];
        
        // Scale coordinates back to original image size
        Rect box = boxes[idx];
        if (ratio_h > ratio_w)
        {
            box.x = box.x / ratio_w;
            box.y = (box.y - (input_h - ratio_w * original_image_height) / 2) / ratio_w;
            box.width = box.width / ratio_w;
            box.height = box.height / ratio_w;
        }
        else
        {
            box.x = (box.x - (input_w - ratio_h * original_image_width) / 2) / ratio_h;
            box.y = box.y / ratio_h;
            box.width = box.width / ratio_h;
            box.height = box.height / ratio_h;
        }
        
        result.bbox = box;
        output.push_back(result);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    postprocess_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Switch buffers for next frame (double buffering)
    current_buffer_idx = 1 - current_buffer_idx;
    
    // Print detailed timing breakdown (only for first few frames)
    static int frame_count = 0;
    if (frame_count < 5) {
        printf("Streaming Postprocess breakdown - Frame %d:\n", frame_count);
        printf("  Memory copy: %.2f ms\n", memcpy_time);
        printf("  Score loop: %.2f ms\n", score_loop_time);
        printf("  NMS: %.2f ms\n", nms_time);
        printf("  Total: %.2f ms\n", postprocess_time);
        printf("  Candidate boxes: %zu\n", boxes.size());
        printf("  Final detections: %zu\n", nms_result.size());
        printf("---\n");
    }
    frame_count++;
}

void YOLOv11::postprocess_start_next_copy() {
    // Start copying the next frame's data while current frame is being processed
    float* next_cpu_buffer = (current_buffer_idx == 0) ? cpu_output_buffer_alt : cpu_output_buffer;
    cudaStream_t next_stream = (current_buffer_idx == 0) ? stream_alt : stream;
    
    // Start async copy for next frame - this will overlap with current frame's processing
    CUDA_CHECK(cudaMemcpyAsync(next_cpu_buffer, gpu_buffers[1], num_detections * (num_classes + 4) * sizeof(float), cudaMemcpyDeviceToHost, next_stream));
}

void YOLOv11::build(std::string onnxPath, nvinfer1::ILogger& logger)
{
    auto builder = createInferBuilder(logger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    IBuilderConfig* config = builder->createBuilderConfig();
    if (isFP16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    bool parsed = parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
    IHostMemory* plan{ builder->buildSerializedNetwork(*network, *config) };

    runtime = createInferRuntime(logger);

    engine = runtime->deserializeCudaEngine(plan->data(), plan->size());

    context = engine->createExecutionContext();

    delete network;
    delete config;
    delete parser;
    delete plan;
}

bool YOLOv11::saveEngine(const std::string& onnxpath)
{
    // Create an engine path from onnx path
    std::string engine_path;
    size_t dotIndex = onnxpath.find_last_of(".");
    if (dotIndex != std::string::npos) {
        engine_path = onnxpath.substr(0, dotIndex) + ".engine";
    }
    else
    {
        return false;
    }

    // Save the engine to the path
    if (engine)
    {
        nvinfer1::IHostMemory* data = engine->serialize();
        std::ofstream file;
        file.open(engine_path, std::ios::binary | std::ios::out);
        if (!file.is_open())
        {
            std::cout << "Create engine file" << engine_path << " failed" << std::endl;
            return 0;
        }
        file.write((const char*)data->data(), data->size());
        file.close();

        delete data;
    }
    return true;
}

void YOLOv11::draw(Mat& image, const vector<Detection>& output)
{
    // Coordinates are now already scaled to original image size in postprocess
    for (int i = 0; i < output.size(); i++)
    {
        auto detection = output[i];
        auto box = detection.bbox;
        auto class_id = detection.class_id;
        auto conf = detection.conf;
        cv::Scalar color = cv::Scalar(COLORS[class_id][0], COLORS[class_id][1], COLORS[class_id][2]);

        rectangle(image, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), color, 3);

        // Detection box text
        string class_string = CLASS_NAMES[class_id] + ' ' + to_string(conf).substr(0, 4);
        Size text_size = getTextSize(class_string, FONT_HERSHEY_DUPLEX, 1, 2, 0);
        Rect text_rect(box.x, box.y - 40, text_size.width + 10, text_size.height + 20);
        rectangle(image, text_rect, color, FILLED);
        putText(image, class_string, Point(box.x + 5, box.y - 10), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0), 2, 0);
    }
}

void YOLOv11::drawTimingInfo(Mat& image)
{
    total_time = preprocess_time + inference_time + postprocess_time;
    
    // Update rolling average arrays
    preprocess_history[history_index] = preprocess_time;
    inference_history[history_index] = inference_time;
    postprocess_history[history_index] = postprocess_time;
    total_history[history_index] = total_time;
    
    // Calculate rolling averages
    float avg_preprocess = 0.0f;
    float avg_inference = 0.0f;
    float avg_postprocess = 0.0f;
    float avg_total = 0.0f;
    
    for (int i = 0; i < ROLLING_WINDOW; i++) {
        avg_preprocess += preprocess_history[i];
        avg_inference += inference_history[i];
        avg_postprocess += postprocess_history[i];
        avg_total += total_history[i];
    }
    
    avg_preprocess /= ROLLING_WINDOW;
    avg_inference /= ROLLING_WINDOW;
    avg_postprocess /= ROLLING_WINDOW;
    avg_total /= ROLLING_WINDOW;
    
    // Update history index (circular buffer)
    history_index = (history_index + 1) % ROLLING_WINDOW;
    
    // Background for timing info
    Rect timing_rect(10, 10, 350, 120);
    rectangle(image, timing_rect, Scalar(0, 0, 0), FILLED);
    rectangle(image, timing_rect, Scalar(255, 255, 255), 2);
    
    // Text properties
    int font = FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.6;
    Scalar text_color = Scalar(0, 255, 0);
    int thickness = 2;
    
    // Display smoothed timing information
    string preprocess_str = "Preprocess: " + to_string(avg_preprocess).substr(0, 5) + " ms";
    string inference_str = "Inference: " + to_string(avg_inference).substr(0, 5) + " ms";
    string postprocess_str = "Postprocess: " + to_string(avg_postprocess).substr(0, 5) + " ms";
    string total_str = "Total: " + to_string(avg_total).substr(0, 5) + " ms";
    string fps_str = "FPS: " + to_string(1000.0f / avg_total).substr(0, 5);
    
    putText(image, preprocess_str, Point(20, 35), font, font_scale, text_color, thickness);
    putText(image, inference_str, Point(20, 55), font, font_scale, text_color, thickness);
    putText(image, postprocess_str, Point(20, 75), font, font_scale, text_color, thickness);
    putText(image, total_str, Point(20, 95), font, font_scale, text_color, thickness);
    putText(image, fps_str, Point(20, 115), font, font_scale, text_color, thickness);
}