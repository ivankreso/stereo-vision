cmake_minimum_required(VERSION 2.8)

string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)
message("Mode: ${CMAKE_BUILD_TYPE}")
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
  set(CMAKE_CXX_FLAGS "-std=c++11 -march=native")
elseif(CMAKE_BUILD_TYPE MATCHES RELEASE)
  set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -fopenmp")
else()
  message(FATAL_ERROR "Unknwon Mode: ${CMAKE_BUILD_TYPE}")
endif()

include_directories(/usr/include/eigen3)
include_directories(/home/kivan/libraries/install/opencv/include)
link_directories(/home/kivan/libraries/install/opencv/lib)

file(GLOB SRC_LIST *.cc)
set(SRC_LIST ${SRC_LIST})
add_subdirectory(../../../core/ libs/core/)
add_subdirectory(../../base/ libs/base/)

set(OPENCV_LIBS opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui)

add_executable(create_rgbd_dataset ${SRC_LIST})
target_link_libraries(create_rgbd_dataset ${OPENCV_LIBS} core recon_base)

