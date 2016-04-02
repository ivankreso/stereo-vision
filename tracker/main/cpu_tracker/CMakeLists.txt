cmake_minimum_required(VERSION 2.8)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to RELEASE")
  set(CMAKE_BUILD_TYPE "release")
endif()

string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)
message("Mode: ${CMAKE_BUILD_TYPE}")
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
  set(CMAKE_CXX_FLAGS "-pedantic -Wall -Wextra -std=c++11 -march=native")
elseif(CMAKE_BUILD_TYPE MATCHES RELEASE)
  set(CMAKE_CXX_FLAGS "-pedantic -Wall -std=c++11 -march=native -fopenmp")
else()
  message(FATAL_ERROR "Unknwon Mode: ${CMAKE_BUILD_TYPE}")
endif()

# force static linking for easy running on other archlinux machines
#set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})
#set(BUILD_SHARED_LIBS OFF)
#set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++ -static-libgfortran")


#include_directories(~/libraries/install/opencv/include/)
#link_directories(~/libraries/install/opencv/lib/)

file(GLOB SRC_LIST *.cc ../../*.cc)
file(GLOB_RECURSE SRC_LIST_LIBVISO ../../extern/libviso2/src/*.cpp)
set(SRC_HELPERS)
set(SRC_LIST ${SRC_LIST} ${SRC_LIST_LIBVISO} ${SRC_HELPERS})

#add_subdirectory(../../../tracker/stereo/ libs/stereo)
#add_subdirectory(../../../tracker/stereo/ libs/stereo)
#add_subdirectory(../../../optimization/bundle_adjustment libs/bundle_adjustment)
#add_subdirectory(../../../stereo_egomotion/base/ libs/egomotion_base)

message(STATUS "${SRC_LIST}")

set(OPENCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_features2d)
#set(BOOST_LIBS boost_program_options boost_serialization)
message(STATUS "OpenCV = ${OPENCV_LIBS}")

add_executable(cpu_tracker ${SRC_LIST})
target_link_libraries(cpu_tracker ${OPENCV_LIBS})

