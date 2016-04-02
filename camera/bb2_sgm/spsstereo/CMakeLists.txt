#Project
cmake_minimum_required(VERSION 2.4)
project(spsstereo)

set(CMAKE_CXX_FLAGS "-std=c++11 -march=native")

# Flags
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3 -msse4.2")
if (APPLE)
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3 -msse4.2 -stdlib=libstdc++")
endif (APPLE)
set(CMAKE_BUILD_TYPE Release)

# SPS-Stereo
file(GLOB SPSSTEREO_SRC_FILES "*.cpp")
add_executable(sps_stereo spsstereo_main.cc ${SPSSTEREO_SRC_FILES})
add_executable(sgm_stereo sgm_stereo_main.cc ${SPSSTEREO_SRC_FILES})
target_link_libraries(sps_stereo png opencv_core opencv_imgproc opencv_highgui)
target_link_libraries(sgm_stereo png opencv_core opencv_imgproc opencv_highgui)
