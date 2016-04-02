cmake_minimum_required(VERSION 2.8)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to RELEASE")
  set(CMAKE_BUILD_TYPE "release")
endif()
string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)
message("Mode: ${CMAKE_BUILD_TYPE}")
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
  set(CMAKE_CXX_FLAGS "-fdiagnostics-color=always -pedantic -Wall -Wno-sign-compare -Wno-unused-function -std=c++11 -march=native")
elseif(CMAKE_BUILD_TYPE MATCHES RELEASE)
  set(CMAKE_CXX_FLAGS "-fdiagnostics-color=always -pedantic -Wall -Wno-sign-compare -Wno-unused-function -std=c++11 -march=native -fopenmp")
else()
  message(FATAL_ERROR "Unknwon Mode: ${CMAKE_BUILD_TYPE}")
endif()

include_directories(/usr/include/eigen3)
include_directories(/home/kivan/libraries/install/include)
link_directories(/home/kivan/libraries/install/lib)

file(GLOB SRC_LIST . *.cc)

add_subdirectory(../../stereo/ libs/stereo)

set(OPENCV_LIBS opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui opencv_features2d
                opencv_cudafeatures2d opencv_xfeatures2d)

message(STATUS "${SRC_LIST}")
message(STATUS "${OPENCV_LIBS}")

add_executable(stereo_tracker ${SRC_LIST})
target_link_libraries(stereo_tracker ${OPENCV_LIBS} tracker_stereo)
