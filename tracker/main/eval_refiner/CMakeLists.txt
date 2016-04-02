# build example:
# in debug/ folder:
# cmake src_dir -DCMAKE_BUILD_TYPE=debug
# in release/ folder:
# cmake src_dir -DCMAKE_BUILD_TYPE=release

cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native")

find_package(OpenCV COMPONENTS opencv_core opencv_imgproc opencv_video opencv_highgui REQUIRED)
include_directories(/usr/include/eigen3/)
file(GLOB SRC_LIST . *.cc)

add_subdirectory(../../mono/ libs/mono)
add_subdirectory(../../refiner/ libs/refiner)

message(STATUS "${SRC_LIST}")
message(STATUS "${OpenCV_LIBS}")

add_executable(eval_refiner ${SRC_LIST})
target_link_libraries(eval_refiner core tracker_mono refiner ${OpenCV_LIBS})

