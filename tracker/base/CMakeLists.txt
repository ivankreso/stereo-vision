# run example: (in build dir)
# in Debug/ folder:
# cmake src_dir -DCMAKE_BUILD_TYPE=debug
# in Release/ folder:
# cmake src_dir -DCMAKE_BUILD_TYPE=release

cmake_minimum_required(VERSION 2.8)


include_directories(/usr/include/eigen3/)

file(GLOB SRC_LIST . *.cc)

#message(STATUS "${SRC_LIST}")
#message(STATUS "${OpenCV_LIBS}")
#message(STATUS "${Boost_LIBRARIES}")

if(NOT TARGET core)
  add_subdirectory(../../core libs/core)
endif()

add_library(tracker_base ${SRC_LIST})
target_link_libraries(tracker_base core)
