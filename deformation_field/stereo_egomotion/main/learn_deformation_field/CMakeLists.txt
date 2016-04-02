cmake_minimum_required(VERSION 2.8)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to DEBUG")
  set(CMAKE_BUILD_TYPE "DEBUG")
endif()

#set(CMAKE_CXX_COMPILER "icpc")
#set(CMAKE_CXX_FLAGS "-std=c++11 -fopenmp")
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -fopenmp")

# includes
include_directories(/usr/local/include/)
include_directories(/usr/include/eigen3/)
include_directories(/usr/local/include/eigen3/)

include_directories(~/libraries/install/ceres-solver/include/)

# libs
link_directories(/usr/local/lib/)
link_directories(~/libraries/install/ceres-solver/lib/)
link_directories(~/libraries/install/ceres-solver/lib64/)
link_directories(/opt/OpenBLAS/lib/)

file(GLOB SRC_LIST *.cc ../../*.cc)

add_subdirectory(../../../tracker/stereo/ libs/stereo)
add_subdirectory(../../../optimization/calibration_bias libs/calibration_bias)
#add_subdirectory(../../../optimization/bundle_adjustment libs/bundle_adjustment)
#add_subdirectory(../../../stereo_odometry/base/ libs/visodom_base)

message(STATUS "${SRC_LIST}")

set(OPENCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_features2d)
set(BOOST_LIBS boost_program_options boost_serialization)
message(STATUS "OpenCV = ${OPENCV_LIBS}")
message(STATUS "Boost = ${BOOST_LIBS}")

add_executable(learn_df ${SRC_LIST})
target_link_libraries(learn_df ${BOOST_LIBS} ${OPENCV_LIBS} tracker_stereo optim_calibration_bias
                      )
