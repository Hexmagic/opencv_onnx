cmake_minimum_required (VERSION 3.8)

project ("ONNX-Debug")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations -Wno-deprecated-declarations")
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
add_executable(native native.cpp  protobuf/opencv-onnx.pb.cc  )
target_link_libraries(native  ${Protobuf_LIBRARIES} ${OpenCV_LIBS})
add_executable(build_in build_in.cpp)
target_link_libraries(build_in ${OpenCV_LIBS})