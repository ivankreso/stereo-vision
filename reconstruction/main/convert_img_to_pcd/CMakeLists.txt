cmake_minimum_required(VERSION 2.8)

string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)
message("Mode: ${CMAKE_BUILD_TYPE}")
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
  set(CMAKE_CXX_FLAGS "-std=c++11 -march=native")
elseif(CMAKE_BUILD_TYPE MATCHES RELEASE)
  set(CMAKE_CXX_FLAGS "-std=c++11 -march=native")
  #set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -fopenmp")
else()
  message(FATAL_ERROR "Unknwon Mode: ${CMAKE_BUILD_TYPE}")
endif()

file(GLOB SRC_LIST *.cc)
set(SRC_LIST ${SRC_LIST})
add_subdirectory(../../../core/ libs/core/)
#add_subdirectory(../../base/ libs/base/)

include_directories(/usr/include/eigen3/)
include_directories(/usr/include/pcl-1.7/)

set(OPENCV_LIBS opencv_core opencv_imgproc opencv_highgui)
set(PCL_LIBS pcl_common pcl_io)
set(BOOST_LIBS boost_system)

add_executable(convert_img_to_pcd ${SRC_LIST})
target_link_libraries(convert_img_to_pcd ${BOOST_LIBS} ${OPENCV_LIBS} ${PCL_LIBS} core)

