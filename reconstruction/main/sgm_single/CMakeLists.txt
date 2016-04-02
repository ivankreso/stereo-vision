cmake_minimum_required(VERSION 2.8)

message("Mode: ${CMAKE_BUILD_TYPE}")
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -std=c++11 -march=native -fopenmp")

file(GLOB SRC_LIST *.cc)
set(SRC_LIST ${SRC_LIST})
add_subdirectory(../../../core/ libs/core/)
add_subdirectory(../../base/ libs/base/)

set(OPENCV_LIBS opencv_core opencv_imgproc opencv_highgui)

add_executable(sgm_single ${SRC_LIST})
target_link_libraries(sgm_single ${OPENCV_LIBS} core recon_base)

