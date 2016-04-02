cmake_minimum_required(VERSION 2.8)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to DEBUG")
  set(CMAKE_BUILD_TYPE "DEBUG")
endif()

#set(CMAKE_CXX_COMPILER "icpc")
#set(CMAKE_CXX_FLAGS "-std=c++11 -fopenmp")
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -fopenmp")

# force static linking for easy running on other archlinux machines
set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++ -static-libgfortran")

# includes
include_directories(/usr/local/include/)
include_directories(/usr/include/eigen3/)
include_directories(/usr/local/include/eigen3/)

# libs
link_directories(/usr/local/lib/)
link_directories(/opt/intel/lib/intel64/)
link_directories(/opt/intel/mkl/lib/intel64/)
link_directories(/home/kivan/Source/ceres-solver-git/build/lib/)
#link_directories(/home/kivan/Source/ceres-solver-1.9.0/build/lib/)
link_directories(/home/kivan/Source/OpenBLAS-0.2.11/)
#link_directories(/usr/local/atlas/lib/)

file(GLOB SRC_LIST *.cc ../../*.cc)
file(GLOB_RECURSE SRC_LIST_LIBVISO ../../extern/libviso2/src/*.cpp)
set(SRC_HELPERS)
set(SRC_LIST ${SRC_LIST} ${SRC_LIST_LIBVISO} ${SRC_HELPERS})

add_subdirectory(../../../tracker/stereo/ libs/stereo)
add_subdirectory(../../../optimization/bundle_adjustment libs/bundle_adjustment)
add_subdirectory(../../../optimization/calibration_bias libs/calibration_bias)
add_subdirectory(../../../stereo_odometry/base/ libs/visodom_base)

message(STATUS "${SRC_LIST}")

set(OPENCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_features2d)
set(BOOST_LIBS boost_program_options)
message(STATUS "OpenCV = ${OPENCV_LIBS}")
message(STATUS "Boost = ${BOOST_LIBS}")

add_executable(visodom ${SRC_LIST})
target_link_libraries(visodom ${BOOST_LIBS} ${OPENCV_LIBS} tracker_stereo visodom_base  optim_bundle_adjustment optim_calibration_bias)
