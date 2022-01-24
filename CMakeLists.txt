# CMakeList.txt: 顶层 CMake 项目文件，在此处执行全局配置
# 并包含子项目。
#
cmake_minimum_required (VERSION 3.8)

project ("ONNX-Debug")
set(OpenCV_DIR "C:/Users/mix/opencv-4.5.5/install/x64/vc16/lib")
find_package(OpenCV REQUIRED)

include_directories(include)
add_executable(main main.cpp)
target_link_libraries(main ${OpenCV_LIBS})