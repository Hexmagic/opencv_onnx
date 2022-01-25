cmake_minimum_required (VERSION 3.8)

project ("ONNX-Debug")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
find_package(OpenCV REQUIRED)
find_package(protobuf REQUIRED)
include_directories(
    /usr/local/include
    ${Protobuf_INCLUDE_DIRS}
    protobuf
    include
    onnx
)
file(GLOB CPPS src/*.cpp)
message(${Protobuf_LIBRARIES})
add_definitions(-DHAVE_PROTOBUF)
add_executable(main main.cpp ${CPPS} protobuf/opencv-onnx.pb.cc )
target_link_libraries(main  ${Protobuf_LIBRARIES} ${OpenCV_LIBS})