cmake_minimum_required(VERSION 2.8)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to DEBUG")
  set(CMAKE_BUILD_TYPE "DEBUG")
endif()

set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -openmp")


set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++ -static-libgfortran")

#find_package(OpenCV REQUIRED)
find_package(OpenCV COMPONENTS opencv_core opencv_imgproc opencv_video opencv_highgui opencv_calib3d REQUIRED)


# setup boost library
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_STATIC_RUNTIME ON)
set(Boost_USE_MULTITHREADED OFF)
find_package(Boost COMPONENTS program_options)
message(STATUS "${Boost_FOUND}")
if(NOT ${Boost_FOUND})
   message(FATAL_ERROR "boost lib not found!")
endif()
include_directories(${Boost_INCLUDE_DIRS})

include_directories(/usr/include/eigen3/)
include_directories(/usr/local/include/eigen3/)
link_directories(/home/kivan/Source/ceres-solver-1.9.0/build/lib/)
#link_directories(/usr/local/atlas/lib/)
link_directories(/home/kivan/Source/xianyi-OpenBLAS-347dded/)

file(GLOB SRC_LIST *.cc ../../*.cc)
file(GLOB_RECURSE SRC_LIST_LIBVISO ../../extern/libviso2/src/*.cpp)
set(SRC_HELPERS)
set(SRC_LIST ${SRC_LIST} ${SRC_LIST_LIBVISO} ${SRC_HELPERS})

add_subdirectory(../../../tracker/stereo/ libs/stereo)
add_subdirectory(../../../optimization/sba libs/sba)

message(STATUS "${SRC_LIST}")
message(STATUS "${OpenCV_LIBS}")
message(STATUS "${Boost_LIBRARIES}")

# doesn't work..
#set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /home/kivan/Projects/vista-stereo/build/lib/ros_sba/release/)
#FIND_LIBRARY(sbaaa libsllddajkkba.so)
#message(STATUS "SBA lib path: ${sba}")
# works but must be specified in main exe cmake config
#link_directories(/home/kivan/Projects/cv-stereo/build/lib/ros_sba/release/)

add_executable(visodom ${SRC_LIST})
target_link_libraries(visodom ${Boost_LIBRARIES} ${OpenCV_LIBS} tracker_stereo refiner optim_sba)
#target_link_libraries(visodom ${Boost_LIBRARIES} ${OpenCV_LIBS} tracker_stereo refiner)
