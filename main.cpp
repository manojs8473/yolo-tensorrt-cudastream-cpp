#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <iostream>
#include <string>
#include "yolov11.h"


bool IsPathExist(const string& path) {
#ifdef _WIN32
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    return (fileAttributes != INVALID_FILE_ATTRIBUTES);
#else
    return (access(path.c_str(), F_OK) == 0);
#endif
}
bool IsFile(const string& path) {
    if (!IsPathExist(path)) {
        printf("%s:%d %s not exist\n", __FILE__, __LINE__, path.c_str());
        return false;
    }

#ifdef _WIN32
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    return ((fileAttributes != INVALID_FILE_ATTRIBUTES) && ((fileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0));
#else
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
#endif
}

/**
 * @brief Setting up Tensorrt logger
*/
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Only output logs with severity greater than warning
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
}logger;

int main(int argc, char** argv)
{
    // Parse command line arguments
    if (argc < 3) {
        printf("Usage: %s <engine_file> <input_path> [confidence_threshold]\n", argv[0]);
        printf("  engine_file: Path to TensorRT engine file\n");
        printf("  input_path: Path to image, video file, or camera index (0, 1, etc.)\n");
        printf("  confidence_threshold: Detection confidence threshold (default: 0.5)\n");
        return -1;
    }
    
    const string engine_file_path{ argv[1] };
    const string path{ argv[2] };
    float conf_threshold = 0.5f;
    
    // Parse optional confidence threshold
    if (argc >= 4) {
        conf_threshold = std::stof(argv[3]);
        printf("Using confidence threshold: %.2f\n", conf_threshold);
    }
    
    vector<string> imagePathList;
    bool                     isVideo{ false };

    // Check if path is a number (camera index)
    bool isCamera = false;
    if (path.length() == 1 && isdigit(path[0])) {
        isVideo = true;
        isCamera = true;
        printf("Using camera input: %s\n", path.c_str());
    }
    else if (IsFile(path))
    {
        string suffix = path.substr(path.find_last_of('.') + 1);
        if (suffix == "jpg" || suffix == "jpeg" || suffix == "png")
        {
            imagePathList.push_back(path);
        }
        else if (suffix == "mp4" || suffix == "avi" || suffix == "m4v" || suffix == "mpeg" || suffix == "mov" || suffix == "mkv" || suffix == "webm")
        {
            isVideo = true;
            printf("Using video file: %s\n", path.c_str());
        }
        else {
            printf("suffix %s is wrong !!!\n", suffix.c_str());
            abort();
        }
    }
    else if (IsPathExist(path))
    {
        glob(path + "/*.jpg", imagePathList);
    }

    // Assume it's a folder, add logic to handle folders
    // init model
    YOLOv11 model(engine_file_path, logger, conf_threshold);

    if (isVideo) {
        cv::VideoCapture cap;
        
        // Initialize camera or video file
        if (isCamera) {
            int camera_id = std::stoi(path);
            cap.open(camera_id);
            printf("Opening camera %d...\n", camera_id);
        } else {
            cap.open(path);
            printf("Opening video file: %s\n", path.c_str());
        }
        
        // Check if video/camera opened successfully
        if (!cap.isOpened()) {
            printf("Error: Could not open %s: %s\n", isCamera ? "camera" : "video file", path.c_str());
            return -1;
        }
        
        printf("%s opened successfully. Starting processing...\n", isCamera ? "Camera" : "Video");
        
        // Statistics tracking variables
        int frame_count = 0;
        float total_preprocess_time = 0.0f;
        float total_inference_time = 0.0f;
        float total_postprocess_time = 0.0f;
        float total_frame_time = 0.0f;

        while (1)
        {
            Mat image;
            cap >> image;

            if (image.empty()) {
                printf("End of video or failed to read frame\n");
                break;
            }

            if (frame_count == 0) {
                printf("Processing first frame (size: %dx%d)\n", image.cols, image.rows);
            }
            
            // Progress indicator every 30 frames
            if (frame_count % 30 == 0) {
                printf("Processing frame %d...\n", frame_count);
            }
            
            vector<Detection> objects;
            
            if (frame_count == 0) printf("Starting preprocess...\n");
            model.preprocess(image);
            
            if (frame_count == 0) printf("Starting inference...\n");
            auto start = std::chrono::system_clock::now();
            model.infer();
            auto end = std::chrono::system_clock::now();

            // Start next frame's memory copy immediately after inference
            // This overlaps the copy with current frame's processing
            model.postprocess_start_next_copy();

            if (frame_count == 0) printf("Starting postprocess...\n");
            model.postprocess(objects);
            
            if (frame_count == 0) printf("Drawing results...\n");
            model.draw(image, objects);
            model.drawTimingInfo(image);

            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
           // printf("cost %2.4lf ms\n", tc);

            // Accumulate statistics
            frame_count++;
            total_preprocess_time += model.getPreprocessTime();
            total_inference_time += model.getInferenceTime();
            total_postprocess_time += model.getPostprocessTime();
            total_frame_time += model.getTotalTime();

            if (frame_count == 0) printf("Showing first frame...\n");
            imshow("prediction", image);
            
            // Check for ESC key press to exit
            int key = waitKey(1) & 0xFF;
            if (key == 27) { // ESC key
                printf("ESC pressed. Exiting...\n");
                break;
            }
            
            if (frame_count == 0) printf("First frame processing completed!\n");
        }

        // Print average timing summary
        if (frame_count > 0) {
            printf("\n=== PERFORMANCE SUMMARY ===\n");
            printf("Total frames processed: %d\n", frame_count);
            printf("Average preprocess time: %.2f ms\n", total_preprocess_time / frame_count);
            printf("Average inference time: %.2f ms\n", total_inference_time / frame_count);
            printf("Average postprocess time: %.2f ms\n", total_postprocess_time / frame_count);
            printf("Average total time: %.2f ms\n", total_frame_time / frame_count);
            printf("Average FPS: %.2f\n", 1000.0f / (total_frame_time / frame_count));
            printf("===========================\n");
        }

        // Release resources
        destroyAllWindows();
        cap.release();
    }
    else {
        // path to folder saves images
        for (const auto& imagePath : imagePathList)
        {
            // open image
            Mat image = imread(imagePath);
            if (image.empty())
            {
                cerr << "Error reading image: " << imagePath << endl;
                continue;
            }

            vector<Detection> objects;
            model.preprocess(image);

            auto start = std::chrono::system_clock::now();
            model.infer();
            auto end = std::chrono::system_clock::now();

            model.postprocess(objects);
            model.draw(image, objects);
            model.drawTimingInfo(image);

            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("cost %2.4lf ms\n", tc);

            model.draw(image, objects);
            imshow("Result", image);

            waitKey(0);
        }
    }

    return 0;
}