# build example:
# in debug/ folder:
# cmake src_dir -DCMAKE_BUILD_TYPE=debug
# in release/ folder:
# cmake src_dir -DCMAKE_BUILD_TYPE=release

cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native")

find_package(OpenCV COMPONENTS opencv_core opencv_imgproc opencv_video opencv_highgui REQUIRED)

find_package(OpenMP)
if(OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(/usr/include/eigen3/)

file(GLOB SRC_LIST . *.cc)

add_subdirectory(../../stereo/ libs/stereo)

message(STATUS "${SRC_LIST}")
message(STATUS "${OpenCV_LIBS}")

add_executable(stereo_tracker ${SRC_LIST})
target_link_libraries(stereo_tracker ${OpenCV_LIBS} tracker_stereo)
