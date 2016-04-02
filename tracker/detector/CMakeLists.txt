# run example: (in build dir)
# in Debug/ folder:
# cmake src_dir -DCMAKE_BUILD_TYPE=debug
# in Release/ folder:
# cmake src_dir -DCMAKE_BUILD_TYPE=release

cmake_minimum_required(VERSION 2.8)

file(GLOB SRC_LIST *.cc)

if(NOT TARGET tracker_base)
  add_subdirectory(../base/ libs/base)
endif()
if(NOT TARGET core)
  add_subdirectory(../../core libs/core)
endif()

add_library(detector ${SRC_LIST})
target_link_libraries(detector core tracker_base)

