cmake_minimum_required(VERSION 3.18)
project(yolov11-tensorrt-cudaStream LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add compile definitions
add_definitions(-DAPI_EXPORTS)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find required packages
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# Platform-specific TensorRT handling
if(WIN32)
    # Windows TensorRT paths
    if(NOT DEFINED TENSORRT_DIR)
        if(DEFINED ENV{TENSORRT_DIR})
            set(TENSORRT_DIR $ENV{TENSORRT_DIR})
        else()
            set(TENSORRT_DIR "C:/TensorRT" CACHE PATH "TensorRT installation directory")
        endif()
    endif()
    
    # Windows library suffixes
    set(TENSORRT_LIB_DIR "${TENSORRT_DIR}/lib")
    set(TENSORRT_LIBS 
        nvinfer.lib
        nvinfer_plugin.lib
        nvparsers.lib
        nvonnxparser.lib
    )
else()
    # Linux TensorRT paths
    if(NOT DEFINED TENSORRT_DIR)
        if(DEFINED ENV{TENSORRT_DIR})
            set(TENSORRT_DIR $ENV{TENSORRT_DIR})
        else()
            set(TENSORRT_DIR "/usr/local/TensorRT" CACHE PATH "TensorRT installation directory")
        endif()
    endif()
    
    # Linux library suffixes
    set(TENSORRT_LIB_DIR "${TENSORRT_DIR}/lib")
    set(TENSORRT_LIBS 
        nvinfer
        nvinfer_plugin
        nvparsers
        nvonnxparser
    )
endif()

# Check if TensorRT exists
if(NOT EXISTS "${TENSORRT_DIR}/include/NvInfer.h")
    message(FATAL_ERROR "TensorRT not found at ${TENSORRT_DIR}. Please set TENSORRT_DIR to your TensorRT installation path")
endif()

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${TENSORRT_DIR}/include)
include_directories(src/)

# Source files
set(SOURCES
    main.cpp
    src/YOLOv11.cpp
    src/preprocess.cu
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Set CUDA properties
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Link directories
link_directories(${TENSORRT_LIB_DIR})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBS}
)

# Platform-specific compiler options
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(WIN32)
        target_compile_options(${PROJECT_NAME} PRIVATE 
            $<$<COMPILE_LANGUAGE:CXX>:/O2>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        )
    else()
        target_compile_options(${PROJECT_NAME} PRIVATE 
            $<$<COMPILE_LANGUAGE:CXX>:-O3>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        )
    endif()
endif()

# Windows-specific settings
if(WIN32)
    # Add Windows-specific definitions if needed
    target_compile_definitions(${PROJECT_NAME} PRIVATE WIN32_LEAN_AND_MEAN)
endif()