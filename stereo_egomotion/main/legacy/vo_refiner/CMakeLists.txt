# run example: (in build dir)
# in Debug/ folder:
# cmake src_dir -DCMAKE_BUILD_TYPE=Debug
# in Release/ folder:
# cmake src_dir -DCMAKE_BUILD_TYPE=Release
#project(StereoOdometry)
cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native")
#find_package(OpenCV REQUIRED)
find_package(OpenCV COMPONENTS opencv_core opencv_imgproc opencv_video opencv_highgui opencv_calib3d REQUIRED)

# setup boost library
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.53.0 COMPONENTS program_options;)
message(STATUS "${Boost_FOUND}")
if(NOT ${Boost_FOUND})
   message(FATAL_ERROR "boost lib not found!")
endif()
include_directories(${Boost_INCLUDE_DIRS})

file(GLOB SRC_LIST *.cc ../../*.cc)
#file(GLOB_RECURSE SRC_LIST *.cc *.c)
file(GLOB_RECURSE SRC_LIST_LIBVISO ../../extern/libviso2/src/*.cpp)
set(SRC_HELPERS)
set(SRC_LIST ${SRC_LIST} ${SRC_LIST_LIBVISO} ${SRC_HELPERS})

add_subdirectory(../../../tracker/stereo/ libs/stereo)
add_subdirectory(../../../tracker/refiner libs/refiner)

message(STATUS "${SRC_LIST}")
message(STATUS "${OpenCV_LIBS}")
message(STATUS "${Boost_LIBRARIES}")

add_executable(visodom ${SRC_LIST})
#target_link_libraries(visodom ${Boost_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(visodom ${Boost_LIBRARIES} ${OpenCV_LIBS} tracker_stereo refiner)
