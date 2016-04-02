cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native")
find_package(OpenCV COMPONENTS opencv_core opencv_highgui opencv_imgproc REQUIRED)

file(GLOB SRC_LIST *.cc)
set(SRC_LIST ${SRC_LIST})

message(STATUS "${OpenCV_LIBS}")

add_subdirectory(../../base/ libs/base/)

add_executable(dense_stereo_eval ${SRC_LIST})
target_link_libraries(dense_stereo_eval ${OpenCV_LIBS} recon_base)
