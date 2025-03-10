cmake_minimum_required(VERSION 3.20)
project(slam_demo)

# 架构判断
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    set(OpenCV_DIR "/opt/homebrew/opt/opencv@4/share/OpenCV")
else()
    set(OpenCV_DIR "/usr/local/opt/opencv@4/share/OpenCV")
endif()

find_package(OpenCV 4 REQUIRED COMPONENTS
        core
        features2d
        highgui
        calib3d
        imgcodecs
)

set(CMAKE_CXX_STANDARD 11)
add_executable(slam_demo src/main.cpp)
target_link_libraries(slam_demo ${OpenCV_LIBS})