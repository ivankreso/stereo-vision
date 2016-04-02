cmake_minimum_required(VERSION 2.8)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to DEBUG")
  set(CMAKE_BUILD_TYPE "debug")
endif()

#string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)
#message("Mode: ${CMAKE_BUILD_TYPE}")
#if(CMAKE_BUILD_TYPE MATCHES DEBUG)
#  set(CMAKE_CXX_FLAGS "-std=c++11 -march=native")
#elseif(CMAKE_BUILD_TYPE MATCHES RELEASE)
#  set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -fopenmp")
#else()
#  message(FATAL_ERROR "Unknwon Mode: ${CMAKE_BUILD_TYPE}")
#endif()

set(CMAKE_CXX_FLAGS_RELEASE "-std=c++11 -Wall -O3 -msse4.2")
set(CMAKE_BUILD_TYPE Release)

# includes
include_directories(/usr/local/include/)

# libs
link_directories(/usr/local/lib/)

file(GLOB SRC_LIST *.cc)

message(STATUS "${SRC_LIST}")

set(OPENCV_LIBS opencv_core opencv_imgproc opencv_calib3d opencv_highgui)
message(STATUS "OpenCV = ${OPENCV_LIBS}")

add_executable(camera_grab ${SRC_LIST})
target_link_libraries(camera_grab ${BOOST_LIBS} ${OPENCV_LIBS} dc1394 png)

