cmake_minimum_required(VERSION 2.8)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to DEBUG")
  set(CMAKE_BUILD_TYPE "DEBUG")
endif()

# cmake -DCMAKE_BUILD_TYPE=release -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc -DCMAKE_C_FLAGS=-mmic -DCMAKE_CXX_FLAGS=-mmic ../../../stereo_odometry/main/vo_batch_debug_phi/
set(CMAKE_CXX_COMPILER "icpc")
set(CMAKE_CXX_FLAGS "-std=c++11 -mmic -fopenmp")
#set(CMAKE_CXX_FLAGS "-std=c++11 -mmic -fopenmp -vec -vec-report6")
#set(CMAKE_CXX_FLAGS "-cxxlib=/usr/linux-k1om-4.7/ -std=c++11 -mmic -openmp")
#set(CMAKE_CXX_FLAGS "-cxxlib=/opt/rh/devtoolset-2/root/usr/lib/gcc/x86_64-redhat-linux/4.8.2/ -std=c++11 -mmic -openmp")

#set(CMAKE_CXX_COMPILER "g++")
#set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -openmp")

# includes
#include_directories(/usr/local/include/)
include_directories(/mnt/ssd/kivan/xeon_phi/libs/src/boost_1_56_0/)
include_directories(/usr/local/include/eigen3/)
include_directories(/mnt/ssd/kivan/xeon_phi/libs/opencv/include/)

# libs
link_directories(/mnt/ssd/kivan/xeon_phi/libs/opencv/lib/)
link_directories(/opt/intel/mkl/lib/mic/)
link_directories(/home/kivan/Source/ceres-solver-1.9.0/build/lib/)
link_directories(/mnt/ssd/kivan/xeon_phi/libs/src/boost_1_56_0/stage/lib/)
#link_directories(/usr/local/atlas/lib/)
#link_directories(/home/kivan/Source/xianyi-OpenBLAS-347dded/)

file(GLOB SRC_LIST *.cc ../../*.cc)
#file(GLOB_RECURSE SRC_LIST_LIBVISO ../../extern/libviso2/src/*.cpp)
#set(SRC_LIST ${SRC_LIST} ${SRC_LIST_LIBVISO} ${SRC_HELPERS})

add_subdirectory(../../../tracker/stereo/ libs/stereo)
#add_subdirectory(../../../optimization/sba libs/sba)
add_subdirectory(../../../stereo_odometry/base/ libs/visodom_base)

message(STATUS "${SRC_LIST}")

set(OPENCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_features2d)
set(BOOST_LIBS boost_program_options)
message(STATUS "OpenCV = ${OPENCV_LIBS}")
message(STATUS "Boost = ${BOOST_LIBS}")

add_executable(visodom ${SRC_LIST})
target_link_libraries(visodom ${BOOST_LIBS} ${OPENCV_LIBS} tracker_stereo visodom_base)
