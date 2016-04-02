# run example: (in build dir)
# in Debug/ folder:
# cmake src_dir -DCMAKE_BUILD_TYPE=debug
# in Release/ folder:
# cmake src_dir -DCMAKE_BUILD_TYPE=release

cmake_minimum_required(VERSION 2.8)

set(CMAKE_BUILD_TYPE release)

set(CMAKE_CXX_FLAGS "-std=c++11 -march=native")

find_package(OpenCV COMPONENTS opencv_core opencv_imgproc opencv_video opencv_highgui REQUIRED)

include_directories("../../base/")
include_directories("../../mono/")
file(GLOB SRC_LIST . *.cc)

message(STATUS "${SRC_LIST}")
message(STATUS "${OpenCV_LIBS}")
message(STATUS "${CMAKE_CXX_FLAGS}")
message(STATUS "${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "${COMPILE_FLAGS}")

add_subdirectory(../../base/ libs/base)
add_subdirectory(../../mono/ libs/mono)

add_executable(opencv_tracker ${SRC_LIST})
target_link_libraries(opencv_tracker libtracker_base libtracker_mono)
