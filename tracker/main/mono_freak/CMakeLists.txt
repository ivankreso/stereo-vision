# build example:
# in debug/ folder:
# cmake src_dir -DCMAKE_BUILD_TYPE=debug
# in release/ folder:
# cmake src_dir -DCMAKE_BUILD_TYPE=release

cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native")

find_package(OpenCV COMPONENTS opencv_core opencv_imgproc opencv_video opencv_highgui REQUIRED)

file(GLOB SRC_LIST . *.cc)

add_subdirectory(../../mono/ libs/mono)

message(STATUS "${SRC_LIST}")
message(STATUS "${OpenCV_LIBS}")

add_executable(eval_tracker ${SRC_LIST})
target_link_libraries(eval_tracker core tracker_mono ${OpenCV_LIBS})

