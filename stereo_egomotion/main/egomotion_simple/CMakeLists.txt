cmake_minimum_required(VERSION 2.8)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to DEBUG")
  set(CMAKE_BUILD_TYPE "debug")
endif()

#set(CMAKE_CXX_COMPILER "icpc")
#set(CMAKE_CXX_FLAGS "-std=c++11 -fopenmp")
#set(CMAKE_CXX_COMPILER "g++")

string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)
message("Mode: ${CMAKE_BUILD_TYPE}")
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
  set(CMAKE_CXX_FLAGS "-pedantic -Wall -std=c++11 -march=native")
elseif(CMAKE_BUILD_TYPE MATCHES RELEASE)
  #set(CMAKE_CXX_FLAGS "-pedantic -Wall -Wextra -std=c++11 -march=native -fopenmp")
  set(CMAKE_CXX_FLAGS "-pedantic -Wall -std=c++11 -march=native -fopenmp")
else()
  message(FATAL_ERROR "Unknwon Mode: ${CMAKE_BUILD_TYPE}")
endif()

# force static linking for easy running on other archlinux machines
#set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})
#set(BUILD_SHARED_LIBS OFF)
#set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++ -static-libgfortran")

# includes
include_directories(/usr/local/include/)
include_directories(/usr/include/eigen3/)
include_directories(/usr/local/include/eigen3/)
# duplicate because of CentOS cmake version
#include_directories(/home/kivan/Source/lib/ceres/include/)
include_directories(~/libraries/install/ceres-solver/include/)

# libs
link_directories(/usr/local/lib/)
link_directories(/opt/intel/lib/intel64/)
link_directories(/opt/intel/mkl/lib/intel64/)
link_directories(~/libraries/install/ceres-solver/lib/)
link_directories(~/libraries/install/ceres-solver/lib64/)
link_directories(/opt/OpenBLAS/lib/)

file(GLOB SRC_LIST *.cc ../../*.cc)

add_subdirectory(../../../tracker/stereo/ libs/stereo)
add_subdirectory(../../../optimization/bundle_adjustment libs/bundle_adjustment)
add_subdirectory(../../../stereo_egomotion/base/ libs/egomotion_base)

message(STATUS "${SRC_LIST}")

set(OPENCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_features2d)
set(BOOST_LIBS boost_program_options boost_serialization)
message(STATUS "OpenCV = ${OPENCV_LIBS}")
message(STATUS "Boost = ${BOOST_LIBS}")

add_executable(egomotion ${SRC_LIST})
target_link_libraries(egomotion ${BOOST_LIBS} ${OPENCV_LIBS} tracker_stereo egomotion_base
                      optim_bundle_adjustment)

