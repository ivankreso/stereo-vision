# run example: (in build dir)
# in Debug/ folder:
# cmake src_dir -DCMAKE_BUILD_TYPE=debug
# in Release/ folder:
# cmake src_dir -DCMAKE_BUILD_TYPE=release

cmake_minimum_required(VERSION 2.8)

set(CMAKE_BUILD_TYPE release)

set(CMAKE_CXX_FLAGS "-std=c++11 -march=native")

find_package(OpenCV COMPONENTS opencv_core opencv_imgproc opencv_calib3d opencv_highgui REQUIRED)

file(GLOB SRC_LIST . *.cc)

add_executable(rectifier ${SRC_LIST})
target_link_libraries(rectifier ${OpenCV_LIBS})
