cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -fopenmp")

# setup boost library
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED OFF)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.53.0 COMPONENTS program_options;)
message(STATUS "${Boost_FOUND}")
if(NOT ${Boost_FOUND})
   message(FATAL_ERROR "boost lib not found!")
endif()

include_directories(/usr/include/eigen3)
include_directories(/home/kivan/libraries/install/opencv/include)
link_directories(/home/kivan/libraries/install/opencv/lib)

file(GLOB SRC_LIST *.cc)
set(SRC_LIST ${SRC_LIST})

message(STATUS "${OpenCV_LIBS}")
message(STATUS "${Boost_LIBRARIES}")

add_subdirectory(../../../core/ libs/core/)
add_subdirectory(../../base/ libs/base/)

set(OPENCV_LIBS opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_calib3d)

add_executable(dense_stereo ${SRC_LIST})
target_link_libraries(dense_stereo ${Boost_LIBRARIES} ${OPENCV_LIBS} core recon_base)
